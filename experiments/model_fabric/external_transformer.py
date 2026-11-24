"""Interpret GPT-2 causal logits or BERT encoder states over a random-access Snapshot."""
import argparse
from contextlib import closing
import hashlib
from itertools import islice
import json
import math
from pathlib import Path
import struct

from .local_seed import LocalFiles, TensorDirectory
from .page_files import PageFiles
from .http_directory import HTTPDirectory


def dot(x,y):
    """Return Σ_i x_i y_i with a Python 3.11 fallback for math.sumprod."""
    if hasattr(math,"sumprod"):
        return math.sumprod(x,y)
    return math.fsum(a*b for a,b in zip(x,y,strict=True))


class Operands:
    def __init__(self, snapshot):
        """Retain addresses (shard,offset,shape) only; weight payload cache = ∅."""
        self.snapshot = snapshot
        self.table = {t["name"]: t for t in snapshot.ground["tensors"]["tensors"]}
        self.bytes = self.requests = self.peak_range = 0

    def rows(self, key, shape, begin=0, end=None):
        """Yield F32 rows i∈[begin,end), from file intervals [base+4ic,base+4(i+1)c)."""
        rows, cols = (1, shape[0]) if len(shape) == 1 else (shape[0], math.prod(shape[1:]))
        stop = rows if end is None else end
        if (self.table[key]["shape"] != list(shape) or self.table[key]["dtype"] != "F32"
                or not 0 <= begin <= stop <= rows or not 0 < cols <= (1 << 20)):
            raise ValueError(f"Incompatible F32 operand {key}")
        shard, header, _, lo, hi, size = self.snapshot.locations[key]
        if hi-lo != 4*rows*cols:
            raise ValueError(f"Invalid byte extent for {key}")
        step = (1 << 20)//cols
        for i in range(begin, stop, step):
            n = min(stop-i, step)
            length = 4*n*cols
            blob, total = self.snapshot.files.read_range(shard, header+lo+4*i*cols, length)
            if total != size or len(blob) != length:
                raise ValueError(f"Changed source extent for {key}")
            self.bytes += length
            self.requests += 1
            self.peak_range = max(self.peak_range, length)
            yield from struct.iter_unpack(f"<{cols}f", blob)

    def vector(self, key, width):
        """Read v[0:d] once for the current normalization or affine operator."""
        return next(self.rows(key, (width,)))

    def embed(self, key, count, width, ids):
        """Gather E[t,:] for the requested token or position addresses t."""
        return [next(self.rows(key, (count,width), i, i+1)) for i in ids]

    def affine(self, name, x, width, transposed=False, bias=True):
        """Evaluate Y=XWᵀ+b, or Y=XW+b for GPT-2's input-major Conv1D layout."""
        depth = len(x[0])
        b = self.vector(name+".bias", width) if bias else [0.]*width
        if not transposed:
            out = [[] for _ in x]
            for j, row in enumerate(self.rows(name+".weight", (width,depth))):
                for result, token in zip(out,x):
                    result.append(b[j]+dot(token,row))
            return out
        out = [list(b) for _ in x]
        rows = self.rows(name+".weight", (depth,width))
        # Each tile contributes X[:,a:b] W[a:b,:]; 32 rows bound decoded scratch.
        offset = 0
        while tile := tuple(islice(rows,32)):
            slices = [token[offset:offset+len(tile)] for token in x]
            for j, col in enumerate(zip(*tile)):
                for result, token in zip(out,slices):
                    result[j] += dot(token,col)
            offset += len(tile)
        return out

    def norm(self, name, x, eps):
        """LN(x)=g⊙(x−μ)/sqrt(mean((x−μ)²)+ε)+b along each token's hidden axis."""
        d = len(x[0])
        gain, bias = self.vector(name+".weight",d), self.vector(name+".bias",d)
        out = []
        for token in x:
            mean = math.fsum(token)/d
            scale = 1/math.sqrt(math.fsum((v-mean)**2 for v in token)/d+eps)
            out.append([(v-mean)*scale*g+b for v,g,b in zip(token,gain,bias)])
        return out


def add(a,b):
    """Pointwise residual Y_it=A_it+B_it, requiring identical shapes."""
    return [[v+w for v,w in zip(x,y,strict=True)] for x,y in zip(a,b,strict=True)]


def attention(q,k,v,heads,causal):
    """Per head, softmax(QKᵀ/√d + mask)V, over j≤i (causal) or all j (encoder)."""
    width = len(q[0])
    if heads <= 0 or width % heads:
        raise ValueError("Incompatible attention geometry")
    d = width//heads
    out = [[] for _ in q]
    for i,query in enumerate(q):
        end = i+1 if causal else len(k)
        for a in range(0,width,d):
            section = slice(a,a+d)
            scores = [dot(query[section],row[section])/math.sqrt(d) for row in k[:end]]
            top = max(scores)
            exps = [math.exp(s-top) for s in scores]
            mass = math.fsum(exps)
            for j in range(a,a+d):
                out[i].append(math.fsum(e*row[j] for e,row in zip(exps,v[:end]))/mass)
    return out


def gpt2(w,c,ids):
    """Pre-LN causal blocks x←x+Attn(LN x), x←x+MLP(LN x), then tied embedding logits."""
    if (c.get("activation_function") != "gelu_new" or c.get("add_cross_attention")
            or c.get("scale_attn_weights",True) is not True or c.get("scale_attn_by_inverse_layer_idx")
            or c.get("tie_word_embeddings",True) is not True or "lm_head.weight" in w.table):
        raise ValueError("Unimplemented GPT-2 operator variant")
    d, vocab = c["n_embd"], c["vocab_size"]
    x = add(w.embed("transformer.wte.weight",vocab,d,ids),
            w.embed("transformer.wpe.weight",c["n_positions"],d,range(len(ids))))
    eps = c["layer_norm_epsilon"]
    for i in range(c["n_layer"]):
        p = f"transformer.h.{i}."
        t = w.affine(p+"attn.c_attn",w.norm(p+"ln_1",x,eps),3*d,True)
        z = attention([r[:d] for r in t],[r[d:2*d] for r in t],[r[2*d:] for r in t],c["n_head"],True)
        x = add(x,w.affine(p+"attn.c_proj",z,d,True))
        y = w.affine(p+"mlp.c_fc",w.norm(p+"ln_2",x,eps),c.get("n_inner") or 4*d,True)
        y = [[.5*v*(1+math.tanh(math.sqrt(2/math.pi)*(v+.044715*v**3))) for v in r] for r in y]
        x = add(x,w.affine(p+"mlp.c_proj",y,d,True))
    last = w.norm("transformer.ln_f",x[-1:],eps)
    return {"logits":w.affine("transformer.wte",last,vocab,bias=False)[0]}


def bert(w,c,ids):
    """Bidirectional post-LN encoder with absolute positions, segment 0 and an all-visible mask."""
    if (c.get("hidden_act") != "gelu" or c.get("position_embedding_type","absolute") != "absolute"
            or c.get("is_decoder") or c.get("add_cross_attention")):
        raise ValueError("Unimplemented BERT operator variant")
    d,eps = c["hidden_size"],c["layer_norm_eps"]
    x = add(w.embed("embeddings.word_embeddings.weight",c["vocab_size"],d,ids),
            w.embed("embeddings.position_embeddings.weight",c["max_position_embeddings"],d,range(len(ids))))
    segment = w.embed("embeddings.token_type_embeddings.weight",c["type_vocab_size"],d,[0])
    x = w.norm("embeddings.LayerNorm",add(x,segment*len(ids)),eps)
    for i in range(c["num_hidden_layers"]):
        p = f"encoder.layer.{i}."
        q,k,v = [w.affine(p+"attention.self."+name,x,d) for name in ("query","key","value")]
        z = w.affine(p+"attention.output.dense",attention(q,k,v,c["num_attention_heads"],False),d)
        x = w.norm(p+"attention.output.LayerNorm",add(x,z),eps)
        y = w.affine(p+"intermediate.dense",x,c["intermediate_size"])
        y = [[.5*v*(1+math.erf(v/math.sqrt(2))) for v in r] for r in y]
        x = w.norm(p+"output.LayerNorm",add(x,w.affine(p+"output.dense",y,d)),eps)
    result = {"last_hidden_state":x}
    if "pooler.dense.weight" in w.table:
        result["pooler_output"] = [math.tanh(v) for v in w.affine("pooler.dense",x[:1],d)[0]]
    return result


def patches(planes, side, kernel, stride, fill):
    """P[y,x,c,i,j]=X[c,stride*y+i−pad,stride*x+j−pad], using fill outside the image."""
    pad = kernel//2
    pitch = side + 2*pad
    padded = []
    for plane in planes:
        frame = [fill]*(pitch*pad)
        for y in range(side):
            frame.extend([fill]*pad + plane[y*side:(y+1)*side] + [fill]*pad)
        frame.extend([fill]*(pitch*pad))
        padded.append(frame)
    for y in range(0,side,stride):
        for x in range(0,side,stride):
            yield tuple(pixel for frame in padded for i in range(kernel)
                        for pixel in frame[(y+i)*pitch+x:(y+i)*pitch+x+kernel])


def conv_bn(w, name, planes, side, width, kernel, stride=1, relu=False):
    """Y_o=γ_o(crosscorrelate(W_o,X)−μ_o)/√(var_o+10⁻⁵)+β_o, optionally ReLU."""
    windows = list(patches(planes,side,kernel,stride,0.))
    norm = name+".normalization."
    mean,var,gain,bias = [w.vector(norm+k,width) for k in ("running_mean","running_var","weight","bias")]
    shape = (width,len(planes),kernel,kernel)
    output = []
    for o,weights in enumerate(w.rows(name+".convolution.weight",shape)):
        scale = gain[o]/math.sqrt(var[o]+1e-5)
        channel = [(dot(weights,window)-mean[o])*scale+bias[o] for window in windows]
        output.append([max(0.,v) for v in channel] if relu else channel)
    return output


def resnet18(w,c,rgb,processor):
    """Basic post-activation block Y=ReLU(BN₂ W₂ ReLU(BN₁ W₁ X)+shortcut(X))."""
    if (c.get("depths") != [2,2,2,2] or c.get("hidden_sizes") != [64,128,256,512]
            or c.get("embedding_size") != 64 or c.get("num_channels") != 3
            or c.get("layer_type") != "basic" or c.get("hidden_act") != "relu"
            or c.get("downsample_in_first_stage") or c.get("downsample_in_bottleneck")
            or len(rgb) != 224*224*3 or processor.get("do_normalize") is not True
            or processor.get("do_rescale",True) is not True or processor.get("rescale_factor",1/255) != 1/255):
        raise ValueError("Unsupported ResNet-18 or RGB preprocessing contract")
    means,stds = processor["image_mean"],processor["image_std"]
    if len(means)!=3 or len(stds)!=3 or not all(math.isfinite(m) for m in means) or not all(
            math.isfinite(s) and s>0 for s in stds):
        raise ValueError("Expected three finite RGB normalization scales and offsets")
    x = [[(v/255.-mean)/std for v in rgb[j::3]] for j,(mean,std) in enumerate(zip(means,stds))]
    x = conv_bn(w,"resnet.embedder.embedder",x,224,64,7,2,True)
    x = [[max(window) for window in patches([plane],112,3,2,-math.inf)] for plane in x]
    side = 56
    for stage,width in enumerate(c["hidden_sizes"]):
        for block in range(c["depths"][stage]):
            name = f"resnet.encoder.stages.{stage}.layers.{block}"
            stride = 2 if stage and block==0 else 1
            shortcut = x if stride==1 else conv_bn(w,name+".shortcut",x,side,width,1,stride)
            y = conv_bn(w,name+".layer.0",x,side,width,3,stride,True)
            side = (side+stride-1)//stride
            y = conv_bn(w,name+".layer.1",y,side,width,3)
            x = [[max(0.,v+s) for v,s in zip(a,b,strict=True)] for a,b in zip(y,shortcut,strict=True)]
    pooled = [math.fsum(plane)/(side*side) for plane in x]
    logits = w.affine("classifier.1",[pooled],len(c["id2label"]))[0]
    return {"logits":logits}


def run(files, ids=None, *, rgb=None):
    """Map a source's typed model graph and token IDs to named observations F_W(ids)."""
    snapshot = TensorDirectory(files)
    c = snapshot.config
    key = (c["model_type"],tuple(c.get("architectures",())))
    adapters = {("gpt2",("GPT2LMHeadModel",)):gpt2, ("bert",("BertModel",)):bert}
    if c.get("quantization_config") or c.get("pruned_heads"):
        raise ValueError(f"Unimplemented model semantics {key}")
    weights = Operands(snapshot)
    if key == ("resnet",("ResNetForImageClassification",)) and rgb is not None and ids is None:
        raw = files.metadata("preprocessor_config.json",1<<20)
        result = resnet18(weights,c,rgb,json.loads(raw))
        inputs = dict(rgb224_sha256=hashlib.sha256(rgb).hexdigest(),processor_sha256=hashlib.sha256(raw).hexdigest())
    elif key in adapters and rgb is None:
        limit = c.get("n_positions",c.get("max_position_embeddings"))
        if not ids or len(ids)>limit or any(type(i) is not int or not 0<=i<c["vocab_size"] for i in ids):
            raise ValueError("Token sequence outside admitted context")
        result = adapters[key](weights,c,ids)
        inputs = dict(tokens=ids)
    else:
        raise ValueError(f"Unimplemented model or input semantics {key}")
    receipt = dict(architecture=key[1][0],**inputs,source=files.authority,revision=files.revision,
                   config_sha256=snapshot.ground["config_sha256"],headers=snapshot.headers,
                   weight_bytes=weights.bytes,weight_requests=weights.requests,max_weight_range=weights.peak_range,
                   observation_sha256=hashlib.sha256(json.dumps(result,allow_nan=False).encode()).hexdigest())
    return result,receipt


def main():
    """Run F_W(ids) with one selected source and publish a digest of its named observations."""
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--snapshot",type=Path)
    source.add_argument("--page")
    source.add_argument("--url", help="Versioned HTTP directory with strong shard ETags and byte ranges")
    inputs = p.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--tokens",type=int,nargs="+")
    inputs.add_argument("--rgb224",type=Path,help="224×224 cropped pixels, row-major interleaved RGB8")
    p.add_argument("--out",type=Path,required=True)
    a = p.parse_args()
    files = LocalFiles(a.snapshot) if a.snapshot else (HTTPDirectory(a.url) if a.url else PageFiles(a.page))
    with closing(files):
        result,receipt = run(files,a.tokens,rgb=a.rgb224.read_bytes() if a.rgb224 else None)
    a.out.write_text(json.dumps(result,allow_nan=False),encoding="utf-8")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()

import unittest
from dataclasses import replace
from observer_closure import form, verify, fixtures, quotient_count, apply


def oracle(actions,x,word,n):
    seen, history = {}, []
    while n and x not in seen:
        seen[x] = len(history)
        history.append(x)
        for a in word:
            x = actions[a][x]
        n -= 1
    if n:
        start = seen[x]
        x = history[start+n%(len(history)-start)]
    return x & 1


class ClosureTests(unittest.TestCase):
    def test_families_and_native_navigation(self):
        maps,o = fixtures()
        for k,rank,classes in ((1,4,16),(2,12,16),(3,25,31)):
            actions = maps[:k]
            g = form(actions,o)
            self.assertEqual(len(g.basis),rank)
            self.assertTrue(verify(g,actions,o))
            self.assertEqual(len({g.encode(x) for x in range(256)}),classes)
            self.assertEqual(quotient_count(actions,o),classes)
            words = [(a,) for a in range(k)] + [tuple(range(k)),tuple(reversed(range(k)))]
            for word in words:
                for n in (0,1,17,10**12+39):
                    powered = g.compile(word,n)
                    for x in range(256):
                        result = (g.observer & apply(powered,g.encode(x))).bit_count() & 1
                        self.assertEqual(result,oracle(actions,x,word,n))
            self.assertEqual(g.navigate(19,(0,),17),oracle(actions,19,(0,),17))

    def test_rank_budget_and_bad_operator(self):
        maps,o = fixtures()
        with self.assertRaisesRegex(OverflowError,"UNRESOLVED"):
            form(maps,o,16)
        g = form(maps[:2],o)
        m = list(g.operators)
        m[0] = (m[0][0]^1,)+m[0][1:]
        self.assertFalse(verify(replace(g,operators=tuple(m)),maps[:2],o))


if __name__ == "__main__":
    unittest.main()

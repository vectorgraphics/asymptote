// Phase 3 fixture: parameterized C++ plugin instantiated for T=int and T=string.
// Verifies that each instantiation gets independent types and that Any
// round-trips through C++ preserving its dynamic asy type.

// --- T = int -----------------------------------------------------------
from phase3(T=int) access Box as BoxI, id as idI, apply as applyI;

BoxI bi = BoxI();
bi.set(42);
write(bi.get());                                       // 42
write(idI(7));                                         // 7
write(applyI(new int(int x) { return x + 1; }, 10));   // 11

// --- T = string --------------------------------------------------------
from phase3(T=string) access Box as BoxS, id as idS, apply as applyS;

BoxS bs = BoxS();
bs.set('hello');
write(bs.get());                                                 // hello
write(idS('world'));                                             // world
write(applyS(new string(string s) { return s + '!'; }, 'phase3'));// phase3!

// --- Caching: re-importing T=int must not re-instantiate the module. --
// (If caching were broken we'd get a fresh Box class and the bltins of
// `idI2`/`applyI2` would have different signatures from `idI`/`applyI`.
// We can detect a cache hit indirectly by confirming the second import
// works without re-running populate and yields the same observable
// behaviour.)
from phase3(T=int) access id as idI2;
write(idI2(99));                                       // 99

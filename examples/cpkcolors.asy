/*
 * Copyright (C) 2003-2005  Miguel, Jmol Development, www.jmol.org
 *
 * Contact: miguel@jmol.org
 *
 *  This library is free software; you can redistribute it and/or
 *  modify it under the terms of the GNU Lesser General Public
 *  License as published by the Free Software Foundation; either
 *  version 2.1 of the License, or (at your option) any later version.
 *
 *  This library is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *  Lesser General Public License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public
 *  License along with this library; if not, write to the Free Software
 *  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 */
string[] Element={
    "Xx", // 0
    "H",  // 1
    "He", // 2
    "Li", // 3
    "Be", // 4
    "B",  // 5
    "C",  // 6
    "N",  // 7
    "O",  // 8
    "F",  // 9
    "Ne", // 10
    "Na", // 11
    "Mg", // 12
    "Al", // 13
    "Si", // 14
    "P",  // 15
    "S",  // 16
    "Cl", // 17
    "Ar", // 18
    "K",  // 19
    "Ca", // 20
    "Sc", // 21
    "Ti", // 22
    "V",  // 23
    "Cr", // 24
    "Mn", // 25
    "Fe", // 26
    "Co", // 27
    "Ni", // 28
    "Cu", // 29
    "Zn", // 30
    "Ga", // 31
    "Ge", // 32
    "As", // 33
    "Se", // 34
    "Br", // 35
    "Kr", // 36
    "Rb", // 37
    "Sr", // 38
    "Y",  // 39
    "Zr", // 40
    "Nb", // 41
    "Mo", // 42
    "Tc", // 43
    "Ru", // 44
    "Rh", // 45
    "Pd", // 46
    "Ag", // 47
    "Cd", // 48
    "In", // 49
    "Sn", // 50
    "Sb", // 51
    "Te", // 52
    "I",  // 53
    "Xe", // 54
    "Cs", // 55
    "Ba", // 56
    "La", // 57
    "Ce", // 58
    "Pr", // 59
    "Nd", // 60
    "Pm", // 61
    "Sm", // 62
    "Eu", // 63
    "Gd", // 64
    "Tb", // 65
    "Dy", // 66
    "Ho", // 67
    "Er", // 68
    "Tm", // 69
    "Yb", // 70
    "Lu", // 71
    "Hf", // 72
    "Ta", // 73
    "W",  // 74
    "Re", // 75
    "Os", // 76
    "Ir", // 77
    "Pt", // 78
    "Au", // 79
    "Hg", // 80
    "Tl", // 81
    "Pb", // 82
    "Bi", // 83
    "Po", // 84
    "At", // 85
    "Rn", // 86
    "Fr", // 87
    "Ra", // 88
    "Ac", // 89
    "Th", // 90
    "Pa", // 91
    "U",  // 92
    "Np", // 93
    "Pu", // 94
    "Am", // 95
    "Cm", // 96
    "Bk", // 97
    "Cf", // 98
    "Es", // 99
    "Fm", // 100
    "Md", // 101
    "No", // 102
    "Lr", // 103
    "Rf", // 104
    "Db", // 105
    "Sg", // 106
    "Bh", // 107
    "Hs", // 108
    "Mt", // 109
    /*
    "Ds", // 110
    "Uuu",// 111
    "Uub",// 112
    "Uut",// 113
    "Uuq",// 114
    "Uup",// 115
    "Uuh",// 116
    "Uus",// 117
    "Uuo",// 118
    */
};

// Default table of CPK atom colors
// (ghemical colors with a few proposed modifications).
string[] Hexcolor={
    "FF1493", // Xx 0
    "FFFFFF", // H  1
    "D9FFFF", // He 2
    "CC80FF", // Li 3
    "C2FF00", // Be 4
    "FFB5B5", // B  5
    "909090", // C  6 - changed from ghemical
    "3050F8", // N  7 - changed from ghemical
    "FF0D0D", // O  8
    "90E050", // F  9 - changed from ghemical
    "B3E3F5", // Ne 10
    "AB5CF2", // Na 11
    "8AFF00", // Mg 12
    "BFA6A6", // Al 13
    "F0C8A0", // Si 14 - changed from ghemical
    "FF8000", // P  15
    "FFFF30", // S  16
    "1FF01F", // Cl 17
    "80D1E3", // Ar 18
    "8F40D4", // K  19
    "3DFF00", // Ca 20
    "E6E6E6", // Sc 21
    "BFC2C7", // Ti 22
    "A6A6AB", // V  23
    "8A99C7", // Cr 24
    "9C7AC7", // Mn 25
    "E06633", // Fe 26 - changed from ghemical
    "F090A0", // Co 27 - changed from ghemical
    "50D050", // Ni 28 - changed from ghemical
    "C88033", // Cu 29 - changed from ghemical
    "7D80B0", // Zn 30
    "C28F8F", // Ga 31
    "668F8F", // Ge 32
    "BD80E3", // As 33
    "FFA100", // Se 34
    "A62929", // Br 35
    "5CB8D1", // Kr 36
    "702EB0", // Rb 37
    "00FF00", // Sr 38
    "94FFFF", // Y  39
    "94E0E0", // Zr 40
    "73C2C9", // Nb 41
    "54B5B5", // Mo 42
    "3B9E9E", // Tc 43
    "248F8F", // Ru 44
    "0A7D8C", // Rh 45
    "006985", // Pd 46
    "C0C0C0", // Ag 47 - changed from ghemical
    "FFD98F", // Cd 48
    "A67573", // In 49
    "668080", // Sn 50
    "9E63B5", // Sb 51
    "D47A00", // Te 52
    "940094", // I  53
    "429EB0", // Xe 54
    "57178F", // Cs 55
    "00C900", // Ba 56
    "70D4FF", // La 57
    "FFFFC7", // Ce 58
    "D9FFC7", // Pr 59
    "C7FFC7", // Nd 60
    "A3FFC7", // Pm 61
    "8FFFC7", // Sm 62
    "61FFC7", // Eu 63
    "45FFC7", // Gd 64
    "30FFC7", // Tb 65
    "1FFFC7", // Dy 66
    "00FF9C", // Ho 67
    "00E675", // Er 68
    "00D452", // Tm 69
    "00BF38", // Yb 70
    "00AB24", // Lu 71
    "4DC2FF", // Hf 72
    "4DA6FF", // Ta 73
    "2194D6", // W  74
    "267DAB", // Re 75
    "266696", // Os 76
    "175487", // Ir 77
    "D0D0E0", // Pt 78 - changed from ghemical
    "FFD123", // Au 79 - changed from ghemical
    "B8B8D0", // Hg 80 - changed from ghemical
    "A6544D", // Tl 81
    "575961", // Pb 82
    "9E4FB5", // Bi 83
    "AB5C00", // Po 84
    "754F45", // At 85
    "428296", // Rn 86
    "420066", // Fr 87
    "007D00", // Ra 88
    "70ABFA", // Ac 89
    "00BAFF", // Th 90
    "00A1FF", // Pa 91
    "008FFF", // U  92
    "0080FF", // Np 93
    "006BFF", // Pu 94
    "545CF2", // Am 95
    "785CE3", // Cm 96
    "8A4FE3", // Bk 97
    "A136D4", // Cf 98
    "B31FD4", // Es 99
    "B31FBA", // Fm 100
    "B30DA6", // Md 101
    "BD0D87", // No 102
    "C70066", // Lr 103
    "CC0059", // Rf 104
    "D1004F", // Db 105
    "D90045", // Sg 106
    "E00038", // Bh 107
    "E6002E", // Hs 108
    "EB0026"  // Mt 109
};



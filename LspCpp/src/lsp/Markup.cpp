//===--- Markup.cpp -----------------------------------------*- C++-*------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://lsp.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "LibLsp/lsp/Markup/Markup.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iterator>
#include <memory>
#include <string>
#include <vector>
#include <boost/algorithm/string.hpp>

namespace lsp {

    /// hexdigit - Return the hexadecimal character for the
    /// given number \p X (which should be less than 16).
    inline char hexdigit(unsigned X, bool LowerCase = false) {
        const char HexChar = LowerCase ? 'a' : 'A';
        return X < 10 ? '0' + X : HexChar + X - 10;
    }

    /// Given an array of c-style strings terminated by a null pointer, construct
    /// a vector of StringRefs representing the same strings without the terminating
    /// null string.
    inline std::vector< std::string_ref> toStringRefArray(const char* const* Strings) {
        std::vector< std::string_ref> Result;
        while (*Strings)
            Result.push_back(*Strings++);
        return Result;
    }

    /// Construct a string ref from a boolean.
    inline  std::string_ref toStringRef(bool B) { return  std::string_ref(B ? "true" : "false"); }

    /// Construct a string ref from an array ref of unsigned chars.
    inline  std::string_ref toStringRef(const std::vector<uint8_t>& Input) {
        return  std::string_ref(Input.begin(), Input.end());
    }

    /// Construct a string ref from an array ref of unsigned chars.
    inline std::vector<uint8_t> arrayRefFromStringRef(const  std::string_ref& Input) {
        return { Input.begin(), Input.end() };
    }

    /// Interpret the given character \p C as a hexadecimal digit and return its
    /// value.
    ///
    /// If \p C is not a valid hex digit, -1U is returned.
    inline unsigned hexDigitValue(char C) {
        struct HexTable {
            unsigned LUT[255] = {};
            constexpr HexTable() {
                // Default initialize everything to invalid.
                for (int i = 0; i < 255; ++i)
                    LUT[i] = ~0U;
                // Initialize `0`-`9`.
                for (int i = 0; i < 10; ++i)
                    LUT['0' + i] = i;
                // Initialize `A`-`F` and `a`-`f`.
                for (int i = 0; i < 6; ++i)
                    LUT['A' + i] = LUT['a' + i] = 10 + i;
            }
        };
        constexpr HexTable Table;
        return Table.LUT[static_cast<unsigned char>(C)];
    }

    /// Checks if character \p C is one of the 10 decimal digits.
    inline bool isDigit(char C) { return C >= '0' && C <= '9'; }

    /// Checks if character \p C is a hexadecimal numeric character.
    inline bool isHexDigit(char C) { return hexDigitValue(C) != ~0U; }

    /// Checks if character \p C is a valid letter as classified by "C" locale.
    inline bool isAlpha(char C) {
        return ('a' <= C && C <= 'z') || ('A' <= C && C <= 'Z');
    }

    /// Checks whether character \p C is either a decimal digit or an uppercase or
    /// lowercase letter as classified by "C" locale.
    inline bool isAlnum(char C) { return isAlpha(C) || isDigit(C); }

    /// Checks whether character \p C is valid ASCII (high bit is zero).
    inline bool isASCII(char C) { return static_cast<unsigned char>(C) <= 127; }

    /// Checks whether all characters in S are ASCII.
    inline bool isASCII(std::string_ref S) {
        for (char C : S)
        {
                if(!isASCII(C))return true;
        }
        return true;
    }

    /// Checks whether character \p C is printable.
    ///
    /// Locale-independent version of the C standard library isprint whose results
    /// may differ on different platforms.
    inline bool isPrint(char C) {
        unsigned char UC = static_cast<unsigned char>(C);
        return (0x20 <= UC) && (UC <= 0x7E);
    }

    /// Checks whether character \p C is whitespace in the "C" locale.
    ///
    /// Locale-independent version of the C standard library isspace.
    inline bool isSpace(char C) {
        return C == ' ' || C == '\f' || C == '\n' || C == '\r' || C == '\t' ||
            C == '\v';
    }

    /// Returns the corresponding lowercase character if \p x is uppercase.
    inline char toLower(char x) {
        if (x >= 'A' && x <= 'Z')
            return x - 'A' + 'a';
        return x;
    }

    /// Returns the corresponding uppercase character if \p x is lowercase.
    inline char toUpper(char x) {
        if (x >= 'a' && x <= 'z')
            return x - 'a' + 'A';
        return x;
    }

    inline std::string utohexstr(uint64_t X, bool LowerCase = false) {
        char Buffer[17];
        char* BufPtr = std::end(Buffer);

        if (X == 0) *--BufPtr = '0';

        while (X) {
            unsigned char Mod = static_cast<unsigned char>(X) & 15;
            *--BufPtr = hexdigit(Mod, LowerCase);
            X >>= 4;
        }

        return std::string(BufPtr, std::end(Buffer));
    }

    /// Convert buffer \p Input to its hexadecimal representation.
    /// The returned string is double the size of \p Input.
    inline std::string toHex( std::string_ref Input, bool LowerCase = false) {
        static const char* const LUT = "0123456789ABCDEF";
        const uint8_t Offset = LowerCase ? 32 : 0;
        size_t Length = Input.size();

        std::string Output;
        Output.reserve(2 * Length);
        for (size_t i = 0; i < Length; ++i) {
            const unsigned char c = Input[i];
            Output.push_back(LUT[c >> 4] | Offset);
            Output.push_back(LUT[c & 15] | Offset);
        }
        return Output;
    }

    inline std::string toHex(std::vector<uint8_t> Input, bool LowerCase = false) {
        return toHex(toStringRef(Input), LowerCase);
    }

    /// Store the binary representation of the two provided values, \p MSB and
    /// \p LSB, that make up the nibbles of a hexadecimal digit. If \p MSB or \p LSB
    /// do not correspond to proper nibbles of a hexadecimal digit, this method
    /// returns false. Otherwise, returns true.
    inline bool tryGetHexFromNibbles(char MSB, char LSB, uint8_t& Hex) {
        unsigned U1 = hexDigitValue(MSB);
        unsigned U2 = hexDigitValue(LSB);
        if (U1 == ~0U || U2 == ~0U)
            return false;

        Hex = static_cast<uint8_t>((U1 << 4) | U2);
        return true;
    }

    /// Return the binary representation of the two provided values, \p MSB and
    /// \p LSB, that make up the nibbles of a hexadecimal digit.
    inline uint8_t hexFromNibbles(char MSB, char LSB) {
        uint8_t Hex = 0;
        bool GotHex = tryGetHexFromNibbles(MSB, LSB, Hex);
        (void)GotHex;
        assert(GotHex && "MSB and/or LSB do not correspond to hex digits");
        return Hex;
    }

    /// Convert hexadecimal string \p Input to its binary representation and store
    /// the result in \p Output. Returns true if the binary representation could be
    /// converted from the hexadecimal string. Returns false if \p Input contains
    /// non-hexadecimal digits. The output string is half the size of \p Input.
    inline bool tryGetFromHex( std::string_ref Input, std::string& Output) {
        if (Input.empty())
            return true;

        Output.reserve((Input.size() + 1) / 2);
        if (Input.size() % 2 == 1) {
            uint8_t Hex = 0;
            if (!tryGetHexFromNibbles('0', Input.front(), Hex))
                return false;

            Output.push_back(Hex);
            Input = Input.drop_front();
        }

        assert(Input.size() % 2 == 0);
        while (!Input.empty()) {
            uint8_t Hex = 0;
            if (!tryGetHexFromNibbles(Input[0], Input[1], Hex))
                return false;

            Output.push_back(Hex);
            Input = Input.drop_front(2);
        }
        return true;
    }

    /// Convert hexadecimal string \p Input to its binary representation.
    /// The return string is half the size of \p Input.
    inline std::string fromHex( std::string_ref Input) {
        std::string Hex;
        bool GotHex = tryGetFromHex(Input, Hex);
        (void)GotHex;
        assert(GotHex && "Input contains non hex digits");
        return Hex;
    }



    inline std::string utostr(uint64_t X, bool isNeg = false) {
        char Buffer[21];
        char* BufPtr = std::end(Buffer);

        if (X == 0) *--BufPtr = '0';  // Handle special case...

        while (X) {
            *--BufPtr = '0' + char(X % 10);
            X /= 10;
        }

        if (isNeg) *--BufPtr = '-';   // Add negative sign...
        return std::string(BufPtr, std::end(Buffer));
    }

    inline std::string itostr(int64_t X) {
        if (X < 0)
            return utostr(static_cast<uint64_t>(1) + ~static_cast<uint64_t>(X), true);
        else
            return utostr(static_cast<uint64_t>(X));
    }

    /// StrInStrNoCase - Portable version of strcasestr.  Locates the first
    /// occurrence of string 's1' in string 's2', ignoring case.  Returns
    /// the offset of s2 in s1 or npos if s2 cannot be found.
     std::string_ref::size_type StrInStrNoCase( std::string_ref s1,  std::string_ref s2);

    /// getToken - This function extracts one token from source, ignoring any
    /// leading characters that appear in the Delimiters string, and ending the
    /// token at any of the characters that appear in the Delimiters string.  If
    /// there are no tokens in the source string, an empty string is returned.
    /// The function returns a pair containing the extracted token and the
    /// remaining tail string.
    std::pair< std::string_ref,  std::string_ref> getToken( std::string_ref Source,
         std::string_ref Delimiters = " \t\n\v\f\r");



    /// Returns the English suffix for an ordinal integer (-st, -nd, -rd, -th).
    inline  std::string_ref getOrdinalSuffix(unsigned Val) {
        // It is critically important that we do this perfectly for
        // user-written sequences with over 100 elements.
        switch (Val % 100) {
        case 11:
        case 12:
        case 13:
            return "th";
        default:
            switch (Val % 10) {
            case 1: return "st";
            case 2: return "nd";
            case 3: return "rd";
            default: return "th";
            }
        }
    }

    namespace detail {

        template <typename IteratorT>
        inline std::string join_impl(IteratorT Begin, IteratorT End,
             std::string_ref Separator, std::input_iterator_tag) {
            std::string S;
            if (Begin == End)
                return S;

            S += (*Begin);
            while (++Begin != End) {
                S += Separator;
                S += (*Begin);
            }
            return S;
        }

        template <typename IteratorT>
        inline std::string join_impl(IteratorT Begin, IteratorT End,
             std::string_ref Separator, std::forward_iterator_tag) {
            std::string S;
            if (Begin == End)
                return S;

            size_t Len = (std::distance(Begin, End) - 1) * Separator.size();
            for (IteratorT I = Begin; I != End; ++I)
                Len += (*I).size();
            S.reserve(Len);
            size_t PrevCapacity = S.capacity();
            (void)PrevCapacity;
            S += (*Begin);
            while (++Begin != End) {
                S += Separator;
                S += (*Begin);
            }
            assert(PrevCapacity == S.capacity() && "String grew during building");
            return S;
        }

        template <typename Sep>
        inline void join_items_impl(std::string& Result, Sep Separator) {}

        template <typename Sep, typename Arg>
        inline void join_items_impl(std::string& Result, Sep Separator,
            const Arg& Item) {
            Result += Item;
        }

        template <typename Sep, typename Arg1, typename... Args>
        inline void join_items_impl(std::string& Result, Sep Separator, const Arg1& A1,
            Args &&... Items) {
            Result += A1;
            Result += Separator;
            join_items_impl(Result, Separator, std::forward<Args>(Items)...);
        }

        inline size_t join_one_item_size(char) { return 1; }
        inline size_t join_one_item_size(const char* S) { return S ? ::strlen(S) : 0; }

        template <typename T> inline size_t join_one_item_size(const T& Str) {
            return Str.size();
        }

        inline size_t join_items_size() { return 0; }

        template <typename A1> inline size_t join_items_size(const A1& A) {
            return join_one_item_size(A);
        }
        template <typename A1, typename... Args>
        inline size_t join_items_size(const A1& A, Args &&... Items) {
            return join_one_item_size(A) + join_items_size(std::forward<Args>(Items)...);
        }

    } // end namespace detail

    /// Joins the strings in the range [Begin, End), adding Separator between
    /// the elements.
    template <typename IteratorT>
    inline std::string join(IteratorT Begin, IteratorT End,  std::string_ref Separator) {
        using tag = typename std::iterator_traits<IteratorT>::iterator_category;
        return detail::join_impl(Begin, End, Separator, tag());
    }

    /// Joins the strings in the range [R.begin(), R.end()), adding Separator
    /// between the elements.
    template <typename Range>
    inline std::string join(Range&& R,  std::string_ref Separator) {
        return join(R.begin(), R.end(), Separator);
    }

    /// Joins the strings in the parameter pack \p Items, adding \p Separator
    /// between the elements.  All arguments must be implicitly convertible to
    /// std::string, or there should be an overload of std::string::operator+=()
    /// that accepts the argument explicitly.
    template <typename Sep, typename... Args>
    inline std::string join_items(Sep Separator, Args &&... Items) {
        std::string Result;
        if (sizeof...(Items) == 0)
            return Result;

        size_t NS = detail::join_one_item_size(Separator);
        size_t NI = detail::join_items_size(std::forward<Args>(Items)...);
        Result.reserve(NI + (sizeof...(Items) - 1) * NS + 1);
        detail::join_items_impl(Result, Separator, std::forward<Args>(Items)...);
        return Result;
    }

    /// A helper class to return the specified delimiter string after the first
    /// invocation of operator  std::string_ref().  Used to generate a comma-separated
    /// list from a loop like so:
    ///
    /// \code
    ///   ListSeparator LS;
    ///   for (auto &I : C)
    ///     OS << LS << I.getName();
    /// \end
    class ListSeparator {
        bool First = true;
         std::string_ref Separator;

    public:
        ListSeparator( std::string_ref Separator = ", ") : Separator(Separator) {}
        operator  std::string_ref() {
            if (First) {
                First = false;
                return {};
            }
            return Separator;
        }
    };

} // end namespace lsp

namespace lsp{

// Is <contents a plausible start to an HTML tag?
// Contents may not be the rest of the line, but it's the rest of the plain
// text, so we expect to see at least the tag name.
bool looksLikeTag(std::string_ref& Contents) {
  if (Contents.empty())
    return false;
  if (Contents.front() == '!' || Contents.front() == '?' ||
      Contents.front() == '/')
    return true;
  // Check the start of the tag name.
  if (!lsp::isAlpha(Contents.front()))
    return false;
  // Drop rest of the tag name, and following whitespace.
  Contents = Contents
                 .drop_while([](char C) {
                   return lsp::isAlnum(C) || C == '-' || C == '_' || C == ':';
                 })
                 .drop_while(lsp::isSpace);
  // The rest of the tag consists of attributes, which have restrictive names.
  // If we hit '=', all bets are off (attribute values can contain anything).
  for (; !Contents.empty(); Contents = Contents.drop_front()) {
    if (lsp::isAlnum(Contents.front()) || lsp::isSpace(Contents.front()))
      continue;
    if (Contents.front() == '>' || Contents.start_with("/>"))
      return true; // May close the tag.
    if (Contents.front() == '=')
      return true; // Don't try to parse attribute values.
    return false;  // Random punctuation means this isn't a tag.
  }
  return true; // Potentially incomplete tag.
}

// Tests whether C should be backslash-escaped in markdown.
// The string being escaped is Before + C + After. This is part of a paragraph.
// StartsLine indicates whether `Before` is the start of the line.
// After may not be everything until the end of the line.
//
// It's always safe to escape punctuation, but want minimal escaping.
// The strategy is to escape the first character of anything that might start
// a markdown grammar construct.
bool needsLeadingEscape(char C, std::string_ref Before,  std::string_ref After,
                        bool StartsLine) {

  auto RulerLength = [&]() -> /*Length*/ unsigned {
    if (!StartsLine || !Before.empty())
      return false;
    std::string_ref A = After.trim_right();
    return std::all_of(A.begin(),A.end(), [C](char D) { return C == D; }) ? 1 + A.size() : 0;
  };
  auto IsBullet = [&]() {
    return StartsLine && Before.empty() &&
           (After.empty() || After.start_with(" "));
  };
  auto SpaceSurrounds = [&]() {
    return (After.empty() || std::isspace(After.front())) &&
           (Before.empty() || std::isspace(Before.back()));
  };

  auto WordSurrounds = [&]() {
    return (!After.empty() && std::isalnum(After.front())) &&
           (!Before.empty() && std::isalnum(Before.back()));
  };

  switch (C) {
  case '\\': // Escaped character.
    return true;
  case '`': // Code block or inline code
    // Any number of backticks can delimit an inline code block that can end
    // anywhere (including on another line). We must escape them all.
    return true;
  case '~': // Code block
    return StartsLine && Before.empty() && After.start_with("~~");
  case '#': { // ATX heading.
    if (!StartsLine || !Before.empty())
      return false;
    std::string_ref& Rest = After.trim_left(C);
    return Rest.empty() || Rest.start_with(" ");
  }
  case ']': // Link or link reference.
    // We escape ] rather than [ here, because it's more constrained:
    //   ](...) is an in-line link
    //   ]: is a link reference
    // The following are only links if the link reference exists:
    //   ] by itself is a shortcut link
    //   ][...] is an out-of-line link
    // Because we never emit link references, we don't need to handle these.
    return After.start_with(":") || After.start_with("(");
  case '=': // Setex heading.
    return RulerLength() > 0;
  case '_': // Horizontal ruler or matched delimiter.
    if (RulerLength() >= 3)
      return true;
    // Not a delimiter if surrounded by space, or inside a word.
    // (The rules at word boundaries are subtle).
    return !(SpaceSurrounds() || WordSurrounds());
  case '-': // Setex heading, horizontal ruler, or bullet.
    if (RulerLength() > 0)
      return true;
    return IsBullet();
  case '+': // Bullet list.
    return IsBullet();
  case '*': // Bullet list, horizontal ruler, or delimiter.
    return IsBullet() || RulerLength() >= 3 || !SpaceSurrounds();
  case '<': // HTML tag (or autolink, which we choose not to escape)
    return looksLikeTag(After);
  case '>': // Quote marker. Needs escaping at start of line.
    return StartsLine && Before.empty();
  case '&': { // HTML entity reference
    auto End = After.find(';');
    if (End == std::string_ref::npos)
      return false;
    std::string_ref Content = After.substr(0, End);
    if (Content.consume_front("#"))
    {
      if (Content.consume_front("x") || Content.consume_front("X"))
      {
          return std::all_of(Content.begin(),Content.end(), lsp::isHexDigit);
      }

      return std::all_of(Content.begin(), Content.end(), [](char c)
      {
              return lsp::isDigit(c);
      });
    }
    return std::all_of(Content.begin(), Content.end(), [](char c)
        {
            return lsp::isAlpha(c);
        });
  }
  case '.': // Numbered list indicator. Escape 12. -> 12\. at start of line.
  case ')':
    return StartsLine && !Before.empty() &&
           std::all_of(Before.begin(), Before.end(), [](char c)
               {
                   return lsp::isDigit(c);
               }) && After.start_with(" ");
  default:
    return false;
  }
}

/// Escape a markdown text block. Ensures the punctuation will not introduce
/// any of the markdown constructs.
 std::string_ref renderText(const std::string_ref& Input, bool StartsLine) {
  std::string_ref R;
  for (unsigned I = 0; I < Input.size(); ++I) {
    if (needsLeadingEscape(Input[I], Input.substr(0, I), Input.substr(I + 1),
                           StartsLine))
      R.push_back('\\');
    R.push_back(Input[I]);
  }
  return R;
}

/// Renders \p Input as an inline block of code in markdown. The returned value
/// is surrounded by backticks and the inner contents are properly escaped.
 std::string_ref renderInlineBlock(const std::string_ref& Input) {
  std::string_ref R;
  // Double all backticks to make sure we don't close the inline block early.
  for (size_t From = 0; From < Input.size();) {
    size_t Next = Input.find("`", From);
    R += Input.substr(From, Next - From);
    if (Next == std::string_ref::npos)
      break;
    R += "``"; // double the found backtick.

    From = Next + 1;
  }
  // If results starts with a backtick, add spaces on both sides. The spaces
  // are ignored by markdown renderers.
  if (std::string_ref(R).start_with("`") || std::string_ref(R).end_with("`"))
    return "` " + std::move(R) + " `";
  // Markdown render should ignore first and last space if both are there. We
  // add an extra pair of spaces in that case to make sure we render what the
  // user intended.
  if (std::string_ref(R).start_with(" ") && std::string_ref(R).end_with(" "))
    return "` " + std::move(R) + " `";
  return "`" + std::move(R) + "`";
}

/// Get marker required for \p Input to represent a markdown codeblock. It
/// consists of at least 3 backticks(`). Although markdown also allows to use
/// tilde(~) for code blocks, they are never used.
 std::string_ref getMarkerForCodeBlock(const std::string_ref& Input) {
  // Count the maximum number of consecutive backticks in \p Input. We need to
  // start and end the code block with more.
  unsigned MaxBackticks = 0;
  unsigned Backticks = 0;
  for (char C : Input) {
    if (C == '`') {
      ++Backticks;
      continue;
    }
    MaxBackticks = std::max(MaxBackticks, Backticks);
    Backticks = 0;
  }
  MaxBackticks = std::max(Backticks, MaxBackticks);
  // Use the corresponding number of backticks to start and end a code block.
  return std::string_ref(/*Repeat=*/std::max(3u, MaxBackticks + 1), '`');
}

 /// SplitString - Split up the specified string according to the specified
/// delimiters, appending the result fragments to the output list.
 void SplitString(const std::string& Source,
                  std::vector<std::string_ref>& OutFragments,
     std::string Delimiters = " \t\n\v\f\r")
{
     boost::split(OutFragments, Source, boost::is_any_of(Delimiters));
}


// Trims the input and concatenates whitespace blocks into a single ` `.
 std::string_ref canonicalizeSpaces(const std::string_ref& Input) {
  std::vector<std::string_ref> Words;
  SplitString(Input, Words);

  return lsp::join(Words, " ");
}


 std::string_ref renderBlocks( std::vector<Block*>&& Children,
     void (Block::* RenderFunc)(std::ostringstream&) const) {
     std::string_ref R;
     std::ostringstream OS(R);

     std::vector<int> v{ 1, 2, 3 };

     // Trim rulers.
     Children.erase(std::remove_if(Children.begin(), Children.end(), [](const Block* C)
          {
                  return C->isRuler();
          }), Children.end());

     bool LastBlockWasRuler = true;
     for (const auto& C : Children) {
         if (C->isRuler() && LastBlockWasRuler)
             continue;
         LastBlockWasRuler = C->isRuler();
         ((*C).*RenderFunc)(OS);
     }

     // Get rid of redundant empty lines introduced in plaintext while imitating
     // padding in markdown.
     std::string_ref AdjustedResult;
     std::string_ref TrimmedText(OS.str());
     TrimmedText = TrimmedText.trim();

     std::copy_if(TrimmedText.begin(), TrimmedText.end(),
         std::back_inserter(AdjustedResult),
         [&TrimmedText](const char& C) {
             return !std::string_ref(TrimmedText.data(),
                 &C - TrimmedText.data() + 1)
                 // We allow at most two newlines.
                 .end_with("\n\n\n");
         });

     return AdjustedResult;
 };
 std::string_ref renderBlocks(const std::vector<std::unique_ptr<Block> >& children,
     void (Block::* renderFunc)(std::ostringstream&) const)
 {
    std::vector<Block*> temp(children.size(), nullptr);
        for(size_t i = 0 ; i < children.size() ; ++i)
        {
        temp[i]=(children[i].get());
        }
    return renderBlocks(std::move(temp), renderFunc);
 }
// Separates two blocks with extra spacing. Note that it might render strangely
// in vscode if the trailing block is a codeblock, see
// https://github.com/microsoft/vscode/issues/88416 for details.
class Ruler : public Block {
public:
  void renderMarkdown(std::ostringstream &OS) const override {
    // Note that we need an extra new line before the ruler, otherwise we might
    // make previous block a title instead of introducing a ruler.
    OS << "\n---\n";
  }
  void renderPlainText(std::ostringstream &OS) const override { OS << '\n'; }
  std::unique_ptr<Block> clone() const override {
    return std::make_unique<Ruler>(*this);
  }
  bool isRuler() const override { return true; }
};

class CodeBlock : public Block {
public:
  void renderMarkdown(std::ostringstream &OS) const override {
    std::string_ref Marker = getMarkerForCodeBlock(Contents);
    // No need to pad from previous blocks, as they should end with a new line.
    OS << Marker << Language << '\n' << Contents << '\n' << Marker << '\n';
  }

  void renderPlainText(std::ostringstream &OS) const override {
    // In plaintext we want one empty line before and after codeblocks.
    OS << '\n' << Contents << "\n\n";
  }

  std::unique_ptr<Block> clone() const override {
    return std::make_unique<CodeBlock>(*this);
  }

  CodeBlock( std::string_ref Contents, std::string_ref Language)
      : Contents(std::move(Contents)), Language(std::move(Language)) {}

private:

  std::string_ref Contents;
  std::string_ref Language;
};

// Inserts two spaces after each `\n` to indent each line. First line is not
// indented.
 std::string_ref indentLines(const std::string_ref& Input) {
  assert(!Input.end_with("\n") && "Input should've been trimmed.");
  std::string_ref IndentedR;
  // We'll add 2 spaces after each new line.
  IndentedR.reserve(Input.size() + Input.count("\n") * 2);
  for (char C : Input) {
    IndentedR += C;
    if (C == '\n')
      IndentedR.append("  ");
  }
  return IndentedR;
}

class Heading : public Paragraph {
public:
  Heading(size_t Level) : Level(Level) {}
  void renderMarkdown(std::ostringstream &OS) const override {
    OS << std::string_ref(Level, '#') << ' ';
    Paragraph::renderMarkdown(OS);
  }

private:
  size_t Level;
};





 std::string_ref Block::asMarkdown() const {
  std::string_ref R;
  std::ostringstream OS(R);
  renderMarkdown(OS);
  return std::string_ref(OS.str()).trim();
}

 std::string_ref Block::asPlainText() const {
  std::string_ref R;
  std::ostringstream OS(R);
  renderPlainText(OS);
  return std::string_ref(OS.str()).trim().c_str();
}

     void Paragraph::renderMarkdown(std::ostringstream& OS) const {
         bool NeedsSpace = false;
         bool HasChunks = false;
         for (auto& C : Chunks) {
             if (C.SpaceBefore || NeedsSpace)
                 OS << " ";
             switch (C.Kind) {
             case Chunk::PlainText:
                 OS << renderText(C.Contents, !HasChunks);
                 break;
             case Chunk::InlineCode:
                 OS << renderInlineBlock(C.Contents);
                 break;
             }
             HasChunks = true;
             NeedsSpace = C.SpaceAfter;
         }
         // Paragraphs are translated into markdown lines, not markdown paragraphs.
         // Therefore it only has a single linebreak afterwards.
         // VSCode requires two spaces at the end of line to start a new one.
         OS << "  \n";
     }

     std::unique_ptr<Block> Paragraph::clone() const {
         return std::make_unique<Paragraph>(*this);
     }

     /// Choose a marker to delimit `Text` from a prioritized list of options.
     /// This is more readable than escaping for plain-text.
     std::string_ref chooseMarker(std::vector<std::string_ref> Options,
         const std::string_ref& Text)
     {
         // Prefer a delimiter whose characters don't appear in the text.
         for (std::string_ref& S : Options)
             if (Text.find_first_of(S) == std::string_ref::npos)
                 return S;
         return Options.front();
     }

     void Paragraph::renderPlainText(std::ostringstream& OS) const {
         bool NeedsSpace = false;
         for (auto& C : Chunks) {
             if (C.SpaceBefore || NeedsSpace)
                 OS << " ";
             std::string_ref Marker = "";
             if (C.Preserve && C.Kind == Chunk::InlineCode)
                 Marker = chooseMarker({ "`", "'", "\"" }, C.Contents);
             OS << Marker << C.Contents << Marker;
             NeedsSpace = C.SpaceAfter;
         }
         OS << '\n';
     }

     void BulletList::renderMarkdown(std::ostringstream& OS) const {
         for (auto& D : Items) {
             // Instead of doing this we might prefer passing Indent to children to get
             // rid of the copies, if it turns out to be a bottleneck.

             OS << "- ";
             OS << indentLines(D.asMarkdown()) << '\n';
         }
         // We need a new line after list to terminate it in markdown.
         OS << '\n';
     }

     void BulletList::renderPlainText(std::ostringstream& OS) const {
         for (auto& D : Items) {
             // Instead of doing this we might prefer passing Indent to children to get
             // rid of the copies, if it turns out to be a bottleneck.
             OS << "- " << indentLines(D.asPlainText()) << '\n';
         }
     }

     Paragraph& Paragraph::appendSpace() {
         if (!Chunks.empty())
             Chunks.back().SpaceAfter = true;
         return *this;
     }

     Paragraph& Paragraph::appendText(const std::string_ref& Text) {
         std::string_ref Norm = canonicalizeSpaces(Text);
         if (Norm.empty())
             return *this;
         Chunks.emplace_back();
         Chunk& C = Chunks.back();
         C.Contents = std::move(Norm);
         C.Kind = Chunk::PlainText;

         C.SpaceBefore = std::isspace(Text.front());
         C.SpaceAfter = std::isspace(Text.back());
         return *this;
     }

     Paragraph& Paragraph::appendCode(const std::string_ref& Code, bool Preserve) {
         bool AdjacentCode =
             !Chunks.empty() && Chunks.back().Kind == Chunk::InlineCode;
         std::string_ref Norm = canonicalizeSpaces(Code);
         if (Norm.empty())
             return *this;
         Chunks.emplace_back();
         Chunk& C = Chunks.back();
         C.Contents = std::move(Norm);
         C.Kind = Chunk::InlineCode;
         C.Preserve = Preserve;
         // Disallow adjacent code spans without spaces, markdown can't render them.
         C.SpaceBefore = AdjacentCode;
         return *this;
     }

     std::unique_ptr<Block> BulletList::clone() const {
         return std::make_unique<BulletList>(*this);
     }

     class Document& BulletList::addItem() {
         Items.emplace_back();
         return Items.back();
     }

     Document& Document::operator=(const Document& Other) {
         Children.clear();
         for (const auto& C : Other.Children)
             Children.push_back(C->clone());
         return *this;
     }

     void Document::append(Document Other) {
         std::move(Other.Children.begin(), Other.Children.end(),
             std::back_inserter(Children));
     }

     Paragraph& Document::addParagraph() {
         Children.push_back(std::make_unique<Paragraph>());
         return *static_cast<Paragraph*>(Children.back().get());
     }

     void Document::addRuler() { Children.push_back(std::make_unique<Ruler>()); }

     void Document::addCodeBlock(std::string_ref Code, std::string_ref Language) {
         Children.emplace_back(
             std::make_unique<CodeBlock>(std::move(Code), std::move(Language)));
     }

     std::string_ref Document::asMarkdown() const {
         return renderBlocks(Children, &Block::renderMarkdown);
     }

     std::string_ref Document::asPlainText() const {
         return renderBlocks(Children, &Block::renderPlainText);
     }

     BulletList& Document::addBulletList() {
         Children.emplace_back(std::make_unique<BulletList>());
         return *static_cast<BulletList*>(Children.back().get());
     }

     Paragraph& Document::addHeading(size_t Level) {
         assert(Level > 0);
         Children.emplace_back(std::make_unique<Heading>(Level));
         return *static_cast<Paragraph*>(Children.back().get());
     }
 };

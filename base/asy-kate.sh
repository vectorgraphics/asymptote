#!/bin/sh
echo  '<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE language SYSTEM "language.dtd">
<!-- based on asy-keywords.el and Highlighting file asymptote.xml by Christoph Hormann-->
<language version="1.0" kateversion="3.2.2" name="asymptote" section="Sources" extensions="*.asy" mimetype="text/x-asymptote" licence="LGPL" author="Carsten Brenner">

<highlighting>
' > asymptote.xml

# 1. Change Name of lists in <\list> <list name="...">
# 2. tail to get rid of the first lines
# 3. building the right line ending
# 4-5. kill linebreaks
# 6. change spaces into <\item><item>
# 7. Undo change (7.) in 'list name'
# 8. do some formatting

cat asy-keywords.el | sed 's/^(.*\-\([^\-]*\)\-.*/\n<list name="\1"><item>/' | tail -14 | sed 's/ ))/<\/item><\/list>/' |  tr '\n' '@' | sed 's/@//g' | sed 's/ /<\/item><item>/g' | sed 's/list<\/item><item>name/list name/g' | sed 's/></>\n</g' >> asymptote.xml

echo '
<contexts>
			<context attribute="Normal Text" lineEndContext="#stay" name="Normal">
				<DetectSpaces />
				<RegExpr attribute="Preprocessor" context="Outscoped" String="#\s*if\s+0" beginRegion="Outscoped" firstNonSpace="true" />
				<DetectChar attribute="Preprocessor" context="Preprocessor" char="#" firstNonSpace="true" />
				<StringDetect attribute="Region Marker" context="Region Marker" String="//BEGIN" beginRegion="Region1" firstNonSpace="true" />
				<StringDetect attribute="Region Marker" context="Region Marker" String="//END" endRegion="Region1" firstNonSpace="true" />
				<keyword attribute="Keyword" context="#stay" String="keyword" />
				<keyword attribute="Extensions" context="#stay" String="extensions" />
				<keyword attribute="Function" context="#stay" String="function" />
				<keyword attribute="Data Type" context="#stay" String="type" />
				<keyword attribute="Constants" context="#stay" String="constants" />
				<keyword attribute="Variable" context="#stay" String="variable" />
				<HlCChar attribute="Char" context="#stay"/>
				<DetectChar attribute="String" context="String" char="&quot;"/>
				<DetectIdentifier />
				<Float attribute="Float" context="#stay">
					<AnyChar String="fF" attribute="Float" context="#stay"/>
				</Float>
				<HlCOct attribute="Octal" context="#stay"/>
				<HlCHex attribute="Hex" context="#stay"/>
				<Int attribute="Decimal" context="#stay">
					<StringDetect attribute="Decimal" context="#stay" String="ULL" insensitive="TRUE"/>
					<StringDetect attribute="Decimal" context="#stay" String="LUL" insensitive="TRUE"/>
					<StringDetect attribute="Decimal" context="#stay" String="LLU" insensitive="TRUE"/>
					<StringDetect attribute="Decimal" context="#stay" String="UL" insensitive="TRUE"/>
					<StringDetect attribute="Decimal" context="#stay" String="LU" insensitive="TRUE"/>
					<StringDetect attribute="Decimal" context="#stay" String="LL" insensitive="TRUE"/>
					<StringDetect attribute="Decimal" context="#stay" String="U" insensitive="TRUE"/>
					<StringDetect attribute="Decimal" context="#stay" String="L" insensitive="TRUE"/>
				</Int>
				<IncludeRules context="##Doxygen" />
				<Detect2Chars attribute="Comment" context="Commentar 1" char="/" char1="/"/>
				<Detect2Chars attribute="Comment" context="Commentar 2" char="/" char1="*" beginRegion="Comment"/>
				<DetectChar attribute="Symbol" context="#stay" char="{" beginRegion="Brace1" />
				<DetectChar attribute="Symbol" context="#stay" char="}" endRegion="Brace1" />
				<AnyChar attribute="Symbol" context="#stay" String=":!%&amp;()+,-/.*&lt;=&gt;?[]{|}~^&#59;"/>
			</context>
			<context attribute="String" lineEndContext="#pop" name="String">
				<LineContinue attribute="String" context="#stay"/>
				<HlCStringChar attribute="String Char" context="#stay"/>
				<DetectChar attribute="String" context="#pop" char="&quot;"/>
			</context>
			<context attribute="Region Marker" lineEndContext="#pop" name="Region Marker">
			</context>
			<context attribute="Comment" lineEndContext="#pop" name="Commentar 1">
				<DetectSpaces />
				<IncludeRules context="##Alerts" />
				<DetectIdentifier />
			</context>
			<context attribute="Comment" lineEndContext="#stay" name="Commentar 2">
				<DetectSpaces />
				<Detect2Chars attribute="Comment" context="#pop" char="*" char1="/" endRegion="Comment"/>
				<IncludeRules context="##Alerts" />
				<DetectIdentifier />
			</context>
			<context attribute="Preprocessor" lineEndContext="#pop" name="Preprocessor">
				<LineContinue attribute="Preprocessor" context="#stay"/>
				<RegExpr attribute="Preprocessor" context="Define" String="define.*((?=\\))"/>
				<RegExpr attribute="Preprocessor" context="#stay" String="define.*"/>
				<RangeDetect attribute="Prep. Lib" context="#stay" char="&quot;" char1="&quot;"/>
				<RangeDetect attribute="Prep. Lib" context="#stay" char="&lt;" char1="&gt;"/>
				<IncludeRules context="##Doxygen" />
				<Detect2Chars attribute="Comment" context="Commentar 1" char="/" char1="/"/>
				<Detect2Chars attribute="Comment" context="Commentar/Preprocessor" char="/" char1="*"/>
			</context>
			<context attribute="Preprocessor" lineEndContext="#pop" name="Define">
				<LineContinue attribute="Preprocessor" context="#stay"/>
			</context>
			<context attribute="Comment" lineEndContext="#stay" name="Commentar/Preprocessor">
				<DetectSpaces />
				<Detect2Chars attribute="Comment" context="#pop" char="*" char1="/" />
				<DetectIdentifier />
			</context>
			<context attribute="Comment" lineEndContext="#stay" name="Outscoped" >
				<DetectSpaces />
				<IncludeRules context="##Alerts" />
				<DetectIdentifier />
				<DetectChar attribute="String" context="String" char="&quot;"/>
				<IncludeRules context="##Doxygen" />
				<Detect2Chars attribute="Comment" context="Commentar 1" char="/" char1="/"/>
				<Detect2Chars attribute="Comment" context="Commentar 2" char="/" char1="*" beginRegion="Comment"/>
				<RegExpr attribute="Comment" context="Outscoped intern" String="#\s*if" beginRegion="Outscoped" firstNonSpace="true" />
				<RegExpr attribute="Preprocessor" context="#pop" String="#\s*(endif|else|elif)" endRegion="Outscoped" firstNonSpace="true" />
			</context>
			<context attribute="Comment" lineEndContext="#stay" name="Outscoped intern">
				<DetectSpaces />
				<IncludeRules context="##Alerts" />
				<DetectIdentifier />
				<DetectChar attribute="String" context="String" char="&quot;"/>
				<IncludeRules context="##Doxygen" />
				<Detect2Chars attribute="Comment" context="Commentar 1" char="/" char1="/"/>
				<Detect2Chars attribute="Comment" context="Commentar 2" char="/" char1="*" beginRegion="Comment"/>
				<RegExpr attribute="Comment" context="Outscoped intern" String="#\s*if" beginRegion="Outscoped" firstNonSpace="true"/>
				<RegExpr attribute="Comment" context="#pop" String="#\s*endif" endRegion="Outscoped" firstNonSpace="true"/>
			</context>
		</contexts>
		<itemDatas>
			<itemData name="Char"         defStyleNum="dsChar"/>
			<itemData name="Comment"      defStyleNum="dsComment"/>
			<itemData name="Data Type"    defStyleNum="dsDataType"/>
			<itemData name="Decimal"      defStyleNum="dsDecVal"/>
			<itemData name="Extensions"   defStyleNum="dsKeyword" color="#0095ff" selColor="#ffffff" bold="1" italic="0"/>
			<itemData name="Float"        defStyleNum="dsFloat"/>
			<itemData name="Function"     defStyleNum="dsFunction" />			
			<itemData name="Hex"          defStyleNum="dsBaseN"/>
			<itemData name="Keyword"      defStyleNum="dsKeyword"/>
			<itemData name="Normal Text"  defStyleNum="dsNormal"/>
			<itemData name="Octal"        defStyleNum="dsBaseN"/>
			<itemData name="Prep. Lib"    defStyleNum="dsOthers"/>
			<itemData name="Preprocessor" defStyleNum="dsOthers"/>
			<itemData name="Region Marker" defStyleNum="dsRegionMarker" />
			<itemData name="String Char"  defStyleNum="dsChar"/>
			<itemData name="String"       defStyleNum="dsString"/>
			<itemData name="Symbol"       defStyleNum="dsNormal"/>
			<itemData name="Variable"     defStyleNum="dsOthers" />
		</itemDatas>
	</highlighting>
	<general>
		<comments>
			<comment name="singleLine" start="//" />
			<comment name="multiLine" start="/*" end="*/" region="Comment"/>
		</comments>
		<keywords casesensitive="1" />
	</general>
  </language>' >> asymptote.xml

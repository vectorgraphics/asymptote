#include "LibLsp/lsp/textDocument/completion.h"
#include "LibLsp/lsp/textDocument/document_symbol.h"
#include "LibLsp/lsp/lsMarkedString.h"
#include "LibLsp/lsp/textDocument/hover.h"
#include "LibLsp/lsp/textDocument/prepareRename.h"
#include <LibLsp/lsp/textDocument/typeHierarchy.h>

#include "LibLsp/lsp/textDocument/semanticHighlighting.h"
#include "LibLsp/lsp/textDocument/SemanticTokens.h"
#include "LibLsp/JsonRpc/json.h"


constexpr unsigned SemanticTokenEncodingSize = 5;

std::string to_string(SemanticTokenType _type)
{
	switch (_type) {

		case ls_namespace: return "namespace";
			/**
			 * Represents a generic type. Acts as a fallback for types which
			 * can"t be mapped to a specific type like class or enum.
			 */
		case ls_type: return "type";
		case ls_class: return "class";
		case ls_enum: return "enum";
		case ls_interface: return "interface";
		case ls_struct: return "struct";
		case ls_typeParameter: return "typeParameter";
		case ls_parameter: return "parameter";
		case ls_variable: return "variable";
		case ls_property: return "property";
		case ls_enumMember: return "enumMember";
		case ls_event: return "event";
		case ls_function: return "function";
		case ls_method: return "method";
		case ls_macro: return "macro";
		case ls_keyword: return "keyword";
		case ls_modifier: return "modifier";
		case ls_comment: return "comment";
		case ls_string: return "string";
		case ls_number: return "number";
		case ls_regexp: return "regexp";
		case ls_operator: return "operator";
		default:
			return  "unknown";
	}
}

unsigned toSemanticTokenType(std::vector<SemanticTokenType>& modifiers)
{
	unsigned encode_type = 0;
	for (auto bit : modifiers) {
		encode_type = encode_type | (0b00000001 << bit);
	}
	return encode_type;
}

std::string to_string(TokenType_JDT _type)
{
	switch (_type)
	{
	case  PACKAGE_JDT:return "namespace";
	case	CLASS_JDT:return "class";
	case	INTERFACE_JDT:return "interface";
	case	ENUM_JDT:return "enum";
	case	ENUM_MEMBER_JDT:return "enumMember";
	case	TYPE_JDT:return "type";
	case	TYPE_PARAMETER_JDT:return "typeParameter";
	case	ANNOTATION_JDT:return "annotation";
	case	ANNOTATION_MEMBER_JDT:return "annotationMember";
	case	METHOD_JDT:return "function";
	case	PROPERTY_JDT:return "property";
	case	VARIABLE_JDT:return "variable";
	case	PARAMETER_JDT:return "parameter";
	}
	return  "unknown";
}

std::string to_string(SemanticTokenModifier modifier)
{
	switch (modifier) {
	case ls_declaration: return  "declaration";
	case ls_definition: return  "definition";
	case ls_readonly: return  "readonly";
	case ls_static: return  "static";
	case ls_deprecated: return  "deprecated";
	case ls_abstract: return  "abstract";
	case ls_async: return  "async";
	case ls_modification: return  "modification";
	case ls_documentation: return  "documentation";
	case ls_defaultLibrary: return  "defaultLibrary";
	default:
		return  "unknown";
	}
}

unsigned toSemanticTokenModifiers(std::vector<SemanticTokenModifier>& modifiers)
{
	unsigned encodedModifiers = 0;
	for (auto bit : modifiers) {
		encodedModifiers = encodedModifiers | (0b00000001 << bit);
	}
	return encodedModifiers;
}


std::string toSemanticTokenType(HighlightingKind_clangD kind) {
	switch (kind) {
	case HighlightingKind_clangD::Variable:
	case HighlightingKind_clangD::LocalVariable:
	case HighlightingKind_clangD::StaticField:
		return "variable";
	case HighlightingKind_clangD::Parameter:
		return "parameter";
	case HighlightingKind_clangD::Function:
		return "function";
	case HighlightingKind_clangD::Method:
		return "method";
	case HighlightingKind_clangD::StaticMethod:
		// FIXME: better method with static modifier?
		return "function";
	case HighlightingKind_clangD::Field:
		return "property";
	case HighlightingKind_clangD::Class:
		return "class";
	case HighlightingKind_clangD::Interface:
		return "interface";
	case HighlightingKind_clangD::Enum:
		return "enum";
	case HighlightingKind_clangD::EnumConstant:
		return "enumMember";
	case HighlightingKind_clangD::Typedef:
	case HighlightingKind_clangD::Type:
		return "type";
	case HighlightingKind_clangD::Unknown:
		return "unknown"; // nonstandard
	case HighlightingKind_clangD::Namespace:
		return "namespace";
	case HighlightingKind_clangD::TemplateParameter:
		return "typeParameter";
	case HighlightingKind_clangD::Concept:
		return "concept"; // nonstandard
	case HighlightingKind_clangD::Primitive:
		return "type";
	case HighlightingKind_clangD::Macro:
		return "macro";
	case HighlightingKind_clangD::InactiveCode:
		return "comment";
	}
	return ("unhandled HighlightingKind_clangD");
}

std::string toSemanticTokenModifier(HighlightingModifier_clangD modifier) {
	switch (modifier) {
	case HighlightingModifier_clangD::Declaration:
		return "declaration";
	case HighlightingModifier_clangD::Deprecated:
		return "deprecated";
	case HighlightingModifier_clangD::Readonly:
		return "readonly";
	case HighlightingModifier_clangD::Static:
		return "static";
	case HighlightingModifier_clangD::Deduced:
		return "deduced"; // nonstandard
	case HighlightingModifier_clangD::Abstract:
		return "abstract";
	case HighlightingModifier_clangD::DependentName:
		return "dependentName"; // nonstandard
	case HighlightingModifier_clangD::DefaultLibrary:
		return "defaultLibrary";
	case HighlightingModifier_clangD::FunctionScope:
		return "functionScope"; // nonstandard
	case HighlightingModifier_clangD::ClassScope:
		return "classScope"; // nonstandard
	case HighlightingModifier_clangD::FileScope:
		return "fileScope"; // nonstandard
	case HighlightingModifier_clangD::GlobalScope:
		return "globalScope"; // nonstandard
	}
	return ("unhandled HighlightingModifier_clangD");
}



bool operator==(const SemanticToken& l, const SemanticToken& r) {
	return std::tie(l.deltaLine, l.deltaStart, l.length, l.tokenType,
		l.tokenModifiers) == std::tie(r.deltaLine, r.deltaStart,
			r.length, r.tokenType,
			r.tokenModifiers);
}

std::vector<int32_t> SemanticTokens::encodeTokens(std::vector<SemanticToken>& tokens)
{
	std::vector<int32_t> result;
	result.reserve(SemanticTokenEncodingSize * tokens.size());
	for (const auto& tok : tokens)
	{
		result.push_back(tok.deltaLine);
		result.push_back(tok.deltaStart);
		result.push_back(tok.length);
		result.push_back(tok.tokenType);
		result.push_back(tok.tokenModifiers);
	}
	assert(result.size() == SemanticTokenEncodingSize * tokens.size());
	return result;
}

void Reflect(Reader& visitor, TextDocumentComplete::Either& value)
{
	if(visitor.IsArray())
	{
		Reflect(visitor, value.first);
	}
	else
	{

		Reflect(visitor, value.second);
	}

}
void Reflect(Reader& visitor, TextDocumentDocumentSymbol::Either& value)
{
	if (visitor.HasMember("location"))
	{
		Reflect(visitor, value.first);
	}
	else
	{
		Reflect(visitor, value.second);
	}
}

void Reflect(Reader& visitor, std::pair<boost::optional<std::string>, boost::optional<lsMarkedString>>& value)
{

	if (!visitor.IsString())
	{
		Reflect(visitor, value.second);
	}
	else
	{
		Reflect(visitor, value.first);
	}
}

void Reflect(Reader& visitor, std::pair<boost::optional<std::string>, boost::optional<MarkupContent>>& value)
{
	if (!visitor.IsString())
	{
		Reflect(visitor, value.second);
	}
	else
	{
		Reflect(visitor, value.first);
	}
}
  void Reflect(Reader& visitor, TextDocumentHover::Either& value)
{
	  JsonReader& reader = dynamic_cast<JsonReader&>(visitor);
	  if (reader.IsArray())
	  {
		  Reflect(visitor, value.first);
	  }
	  else if(reader.m_->IsObject())
	  {
		  Reflect(visitor, value.second);
	  }
}

   void  Reflect(Reader& visitor, TextDocumentPrepareRenameResult& value)
{
	  if (visitor.HasMember("placeholder"))
	  {
		  Reflect(visitor, value.second);
	  }
	  else
	  {
		  Reflect(visitor, value.first);
	  }
}

  namespace
	  RefactorProposalUtility
  {
	    const char* APPLY_REFACTORING_COMMAND_ID = "java.action.applyRefactoringCommand";
	    const char* EXTRACT_VARIABLE_ALL_OCCURRENCE_COMMAND = "extractVariableAllOccurrence";
	    const char* EXTRACT_VARIABLE_COMMAND = "extractVariable";
	    const char* EXTRACT_CONSTANT_COMMAND = "extractConstant";
	    const char* EXTRACT_METHOD_COMMAND = "extractMethod";
	    const char* EXTRACT_FIELD_COMMAND = "extractField";
	    const char* CONVERT_VARIABLE_TO_FIELD_COMMAND = "convertVariableToField";
	    const char* MOVE_FILE_COMMAND = "moveFile";
	    const char* MOVE_INSTANCE_METHOD_COMMAND = "moveInstanceMethod";
	    const char* MOVE_STATIC_MEMBER_COMMAND = "moveStaticMember";
	    const char* MOVE_TYPE_COMMAND = "moveType";
  };
  namespace  QuickAssistProcessor {

	   const char* SPLIT_JOIN_VARIABLE_DECLARATION_ID = "org.eclipse.jdt.ls.correction.splitJoinVariableDeclaration.assist"; //$NON-NLS-1$
	   const char* CONVERT_FOR_LOOP_ID = "org.eclipse.jdt.ls.correction.convertForLoop.assist"; //$NON-NLS-1$
	   const char* ASSIGN_TO_LOCAL_ID = "org.eclipse.jdt.ls.correction.assignToLocal.assist"; //$NON-NLS-1$
	   const char* ASSIGN_TO_FIELD_ID = "org.eclipse.jdt.ls.correction.assignToField.assist"; //$NON-NLS-1$
	   const char* ASSIGN_PARAM_TO_FIELD_ID = "org.eclipse.jdt.ls.correction.assignParamToField.assist"; //$NON-NLS-1$
	   const char* ASSIGN_ALL_PARAMS_TO_NEW_FIELDS_ID = "org.eclipse.jdt.ls.correction.assignAllParamsToNewFields.assist"; //$NON-NLS-1$
	   const char* ADD_BLOCK_ID = "org.eclipse.jdt.ls.correction.addBlock.assist"; //$NON-NLS-1$
	   const char* EXTRACT_LOCAL_ID = "org.eclipse.jdt.ls.correction.extractLocal.assist"; //$NON-NLS-1$
	   const char* EXTRACT_LOCAL_NOT_REPLACE_ID = "org.eclipse.jdt.ls.correction.extractLocalNotReplaceOccurrences.assist"; //$NON-NLS-1$
	   const char* EXTRACT_CONSTANT_ID = "org.eclipse.jdt.ls.correction.extractConstant.assist"; //$NON-NLS-1$
	   const char* INLINE_LOCAL_ID = "org.eclipse.jdt.ls.correction.inlineLocal.assist"; //$NON-NLS-1$
	   const char* CONVERT_LOCAL_TO_FIELD_ID = "org.eclipse.jdt.ls.correction.convertLocalToField.assist"; //$NON-NLS-1$
	   const char* CONVERT_ANONYMOUS_TO_LOCAL_ID = "org.eclipse.jdt.ls.correction.convertAnonymousToLocal.assist"; //$NON-NLS-1$
	   const char* CONVERT_TO_STRING_BUFFER_ID = "org.eclipse.jdt.ls.correction.convertToStringBuffer.assist"; //$NON-NLS-1$
	   const char* CONVERT_TO_MESSAGE_FORMAT_ID = "org.eclipse.jdt.ls.correction.convertToMessageFormat.assist"; //$NON-NLS-1$;
	   const char* EXTRACT_METHOD_INPLACE_ID = "org.eclipse.jdt.ls.correction.extractMethodInplace.assist"; //$NON-NLS-1$;

	   const char* CONVERT_ANONYMOUS_CLASS_TO_NESTED_COMMAND = "convertAnonymousClassToNestedCommand";
  };

  void Reflect(Reader& reader, TypeHierarchyDirection& value) {
	  if (!reader.IsString())
	  {
		  value = TypeHierarchyDirection::Both;
		  return;
	  }
	  std::string v = reader.GetString();
	  if (v == "Children")
		  value = TypeHierarchyDirection::Both;
	  else if (v == "Parents")
		  value = TypeHierarchyDirection::Parents;
	  else if (v == "Both")
		  value = TypeHierarchyDirection::Both;
  }


  void Reflect(Writer& writer, TypeHierarchyDirection& value) {
	  switch (value)
	  {
	  case TypeHierarchyDirection::Children:
		  writer.String("Children");
		  break;
	  case TypeHierarchyDirection::Parents:
		  writer.String("Parents");
		  break;
	  case TypeHierarchyDirection::Both:
		  writer.String("Both");
		  break;
	  }
  }

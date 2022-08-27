#include "LibLsp/lsp/ProtocolJsonHandler.h"
#include "LibLsp/lsp/general/initialize.h"
#include "LibLsp/lsp/general/shutdown.h"
#include "LibLsp/lsp/textDocument/code_action.h"
#include "LibLsp/lsp/textDocument/code_lens.h"
#include "LibLsp/lsp/textDocument/completion.h"


#include "LibLsp/lsp/textDocument/did_close.h"

#include "LibLsp/lsp/textDocument/highlight.h"
#include "LibLsp/lsp/textDocument/document_link.h"
#include "LibLsp/lsp/textDocument/formatting.h"
#include "LibLsp/lsp/textDocument/hover.h"
#include "LibLsp/lsp/textDocument/implementation.h"
#include "LibLsp/lsp/textDocument/range_formatting.h"
#include "LibLsp/lsp/textDocument/references.h"
#include "LibLsp/lsp/textDocument/rename.h"
#include "LibLsp/lsp/textDocument/signature_help.h"
#include "LibLsp/lsp/textDocument/type_definition.h"
#include "LibLsp/lsp/workspace/symbol.h"
#include "LibLsp/lsp/textDocument/typeHierarchy.h"
#include "LibLsp/lsp/out_list.h"
#include "LibLsp/lsp/extention/jdtls/codeActionResult.h"
#include "LibLsp/lsp/textDocument/declaration_definition.h"
#include "LibLsp/lsp/textDocument/resolveCompletionItem.h"
#include "LibLsp/lsp/textDocument/resolveCodeLens.h"
#include "LibLsp/lsp/textDocument/colorPresentation.h"
#include "LibLsp/lsp/textDocument/foldingRange.h"
#include "LibLsp/lsp/textDocument/prepareRename.h"
#include "LibLsp/lsp/textDocument/resolveTypeHierarchy.h"
#include "LibLsp/lsp/textDocument/callHierarchy.h"
#include "LibLsp/lsp/textDocument/selectionRange.h"
#include "LibLsp/lsp/extention/jdtls/classFileContents.h"
#include "LibLsp/lsp/extention/jdtls/buildWorkspace.h"
#include "LibLsp/lsp/extention/jdtls/listOverridableMethods.h"
#include "LibLsp/lsp/extention/jdtls/addOverridableMethods.h"
#include "LibLsp/lsp/extention/jdtls/checkHashCodeEqualsStatus.h"
#include "LibLsp/lsp/extention/jdtls/checkConstructorsStatus.h"
#include "LibLsp/lsp/extention/jdtls/checkDelegateMethodsStatus.h"
#include "LibLsp/lsp/extention/jdtls/checkToStringStatus.h"
#include "LibLsp/lsp/extention/jdtls/executeCommand.h"
#include "LibLsp/lsp/extention/jdtls/findLinks.h"
#include "LibLsp/lsp/extention/jdtls/generateAccessors.h"
#include "LibLsp/lsp/extention/jdtls/generateConstructors.h"
#include "LibLsp/lsp/extention/jdtls/generateDelegateMethods.h"
#include "LibLsp/lsp/extention/jdtls/generateHashCodeEquals.h"
#include "LibLsp/lsp/extention/jdtls/generateToString.h"
#include "LibLsp/lsp/extention/jdtls/getMoveDestinations.h"
#include "LibLsp/lsp/extention/jdtls/Move.h"
#include "LibLsp/lsp/extention/jdtls/organizeImports.h"
#include "LibLsp/lsp/general/exit.h"
#include "LibLsp/lsp/general/initialized.h"
#include "LibLsp/lsp/extention/jdtls/projectConfigurationUpdate.h"
#include "LibLsp/lsp/textDocument/did_change.h"
#include "LibLsp/lsp/textDocument/did_open.h"
#include "LibLsp/lsp/textDocument/did_save.h"
#include "LibLsp/lsp/textDocument/publishDiagnostics.h"
#include "LibLsp/lsp/textDocument/willSave.h"

#include "LibLsp/lsp/workspace/didChangeWorkspaceFolders.h"
#include "LibLsp/lsp/workspace/did_change_configuration.h"
#include "LibLsp/lsp/workspace/did_change_watched_files.h"
#include "LibLsp/lsp/windows/MessageNotify.h"
#include "LibLsp/lsp/language/language.h"
#include "LibLsp/lsp/client/registerCapability.h"
#include "LibLsp/lsp/client/unregisterCapability.h"
#include "LibLsp/JsonRpc/Cancellation.h"
#include "LibLsp/lsp/textDocument/didRenameFiles.h"
#include "LibLsp/lsp/textDocument/semanticHighlighting.h"
#include "LibLsp/lsp/workspace/configuration.h"


void AddStadardResponseJsonRpcMethod(MessageJsonHandler& handler)
{

        handler.method2response[td_initialize::request::kMethodInfo] = [](Reader& visitor)
        {
                if(visitor.HasMember("error"))
                 return         Rsp_Error::ReflectReader(visitor);

                return td_initialize::response::ReflectReader(visitor);
        };

        handler.method2response[td_shutdown::request::kMethodInfo] = [](Reader& visitor)
        {
                if (visitor.HasMember("error"))
                        return  Rsp_Error::ReflectReader(visitor);
                return td_shutdown::response::ReflectReader(visitor);
        };
        handler.method2response[td_codeAction::request::kMethodInfo] = [](Reader& visitor)
        {
                if (visitor.HasMember("error"))
                        return  Rsp_Error::ReflectReader(visitor);

                return td_codeAction::response::ReflectReader(visitor);
        };
        handler.method2response[td_codeLens::request::kMethodInfo] = [](Reader& visitor)
        {
                if (visitor.HasMember("error"))
                        return  Rsp_Error::ReflectReader(visitor);
                return td_codeLens::response::ReflectReader(visitor);
        };
        handler.method2response[td_completion::request::kMethodInfo] = [](Reader& visitor)
        {
                if (visitor.HasMember("error"))
                        return  Rsp_Error::ReflectReader(visitor);
                return td_completion::response::ReflectReader(visitor);
        };

        handler.method2response[td_definition::request::kMethodInfo] = [](Reader& visitor)
        {
                if (visitor.HasMember("error"))
                        return  Rsp_Error::ReflectReader(visitor);
                return td_definition::response::ReflectReader(visitor);
        };
        handler.method2response[td_declaration::request::kMethodInfo] = [](Reader& visitor)
        {
                if (visitor.HasMember("error"))
                        return  Rsp_Error::ReflectReader(visitor);
                return td_declaration::response::ReflectReader(visitor);
        };
        handler.method2response[td_willSaveWaitUntil::request::kMethodInfo] = [](Reader& visitor)
        {
                if (visitor.HasMember("error"))
                        return  Rsp_Error::ReflectReader(visitor);
                return td_willSaveWaitUntil::response::ReflectReader(visitor);
        };

        handler.method2response[td_highlight::request::kMethodInfo] = [](Reader& visitor)
        {
                if (visitor.HasMember("error"))
                        return  Rsp_Error::ReflectReader(visitor);
                return td_highlight::response::ReflectReader(visitor);
        };

        handler.method2response[td_links::request::kMethodInfo] = [](Reader& visitor)
        {
                if (visitor.HasMember("error"))
                        return  Rsp_Error::ReflectReader(visitor);
                return td_links::response::ReflectReader(visitor);
        };

        handler.method2response[td_linkResolve::request::kMethodInfo] = [](Reader& visitor)
        {
                if (visitor.HasMember("error"))
                        return  Rsp_Error::ReflectReader(visitor);
                return td_linkResolve::response::ReflectReader(visitor);
        };

        handler.method2response[td_symbol::request::kMethodInfo] = [](Reader& visitor)
        {
                if (visitor.HasMember("error"))
                        return  Rsp_Error::ReflectReader(visitor);
                return td_symbol::response::ReflectReader(visitor);
        };

        handler.method2response[td_formatting::request::kMethodInfo] = [](Reader& visitor)
        {
                if (visitor.HasMember("error"))
                        return  Rsp_Error::ReflectReader(visitor);
                return td_formatting::response::ReflectReader(visitor);
        };

        handler.method2response[td_hover::request::kMethodInfo] = [](Reader& visitor)
        {
                if (visitor.HasMember("error"))
                        return  Rsp_Error::ReflectReader(visitor);
                return td_hover::response::ReflectReader(visitor);

        };

        handler.method2response[td_implementation::request::kMethodInfo] = [](Reader& visitor)
        {
                if (visitor.HasMember("error"))
                        return  Rsp_Error::ReflectReader(visitor);
                return td_implementation::response::ReflectReader(visitor);
        };

        handler.method2response[td_rangeFormatting::request::kMethodInfo] = [](Reader& visitor)
        {
                if (visitor.HasMember("error"))
                        return  Rsp_Error::ReflectReader(visitor);
                return td_rangeFormatting::response::ReflectReader(visitor);
        };

        handler.method2response[td_references::request::kMethodInfo] = [](Reader& visitor)
        {
                if (visitor.HasMember("error"))
                        return  Rsp_Error::ReflectReader(visitor);
                return td_references::response::ReflectReader(visitor);
        };

        handler.method2response[td_rename::request::kMethodInfo] = [](Reader& visitor)
        {
                if (visitor.HasMember("error"))
                        return  Rsp_Error::ReflectReader(visitor);
                return td_rename::response::ReflectReader(visitor);
        };


        handler.method2response[td_signatureHelp::request::kMethodInfo] = [](Reader& visitor)
        {
                if (visitor.HasMember("error"))
                        return  Rsp_Error::ReflectReader(visitor);
                return td_signatureHelp::response::ReflectReader(visitor);
        };

        handler.method2response[td_typeDefinition::request::kMethodInfo] = [](Reader& visitor)
        {
                if (visitor.HasMember("error"))
                        return  Rsp_Error::ReflectReader(visitor);
                return td_typeDefinition::response::ReflectReader(visitor);
        };

        handler.method2response[wp_executeCommand::request::kMethodInfo] = [](Reader& visitor)
        {
                if (visitor.HasMember("error"))
                        return  Rsp_Error::ReflectReader(visitor);
                return wp_executeCommand::response::ReflectReader(visitor);
        };

        handler.method2response[wp_symbol::request::kMethodInfo] = [](Reader& visitor)
        {
                if (visitor.HasMember("error"))
                        return  Rsp_Error::ReflectReader(visitor);
                return wp_symbol::response::ReflectReader(visitor);
        };
        handler.method2response[td_typeHierarchy::request::kMethodInfo] = [](Reader& visitor)
        {
                if (visitor.HasMember("error"))
                        return  Rsp_Error::ReflectReader(visitor);
                return td_typeHierarchy::response::ReflectReader(visitor);
        };
        handler.method2response[completionItem_resolve::request::kMethodInfo] = [](Reader& visitor)
        {
                if (visitor.HasMember("error"))
                        return  Rsp_Error::ReflectReader(visitor);
                return completionItem_resolve::response::ReflectReader(visitor);
        };

        handler.method2response[codeLens_resolve::request::kMethodInfo] = [](Reader& visitor)
        {
                if (visitor.HasMember("error"))
                        return  Rsp_Error::ReflectReader(visitor);

                return codeLens_resolve::response::ReflectReader(visitor);

        };

        handler.method2response[td_colorPresentation::request::kMethodInfo] = [](Reader& visitor)
        {
                if (visitor.HasMember("error"))
                        return  Rsp_Error::ReflectReader(visitor);

                return td_colorPresentation::response::ReflectReader(visitor);

        };
        handler.method2response[td_documentColor::request::kMethodInfo] = [](Reader& visitor)
        {
                if (visitor.HasMember("error"))
                        return  Rsp_Error::ReflectReader(visitor);

                return td_documentColor::response::ReflectReader(visitor);

        };
        handler.method2response[td_foldingRange::request::kMethodInfo] = [](Reader& visitor)
        {
                if (visitor.HasMember("error"))
                        return  Rsp_Error::ReflectReader(visitor);

                return td_foldingRange::response::ReflectReader(visitor);

        };
        handler.method2response[td_prepareRename::request::kMethodInfo] = [](Reader& visitor)
        {
                if (visitor.HasMember("error"))
                        return  Rsp_Error::ReflectReader(visitor);

                return td_prepareRename::response::ReflectReader(visitor);

        };
        handler.method2response[typeHierarchy_resolve::request::kMethodInfo] = [](Reader& visitor)
        {
                if (visitor.HasMember("error"))
                        return  Rsp_Error::ReflectReader(visitor);

                return typeHierarchy_resolve::response::ReflectReader(visitor);

        };

        handler.method2response[td_selectionRange::request::kMethodInfo] = [](Reader& visitor)
        {
                if (visitor.HasMember("error"))
                        return  Rsp_Error::ReflectReader(visitor);

                return td_selectionRange::response::ReflectReader(visitor);

        };
        handler.method2response[td_didRenameFiles::request::kMethodInfo] = [](Reader& visitor)
        {
                if (visitor.HasMember("error"))
                        return  Rsp_Error::ReflectReader(visitor);

                return td_didRenameFiles::response::ReflectReader(visitor);

        };
        handler.method2response[td_willRenameFiles::request::kMethodInfo] = [](Reader& visitor)
        {
                if (visitor.HasMember("error"))
                        return  Rsp_Error::ReflectReader(visitor);

                return td_willRenameFiles::response::ReflectReader(visitor);

        };

}


void AddJavaExtentionResponseJsonRpcMethod(MessageJsonHandler& handler)
{
        handler.method2response[java_classFileContents::request::kMethodInfo] = [](Reader& visitor)
        {
                if (visitor.HasMember("error"))
                        return  Rsp_Error::ReflectReader(visitor);

                return java_classFileContents::response::ReflectReader(visitor);
        };
        handler.method2response[java_buildWorkspace::request::kMethodInfo] = [](Reader& visitor)
        {
                if (visitor.HasMember("error"))
                        return  Rsp_Error::ReflectReader(visitor);

                return java_buildWorkspace::response::ReflectReader(visitor);
        };
        handler.method2response[java_listOverridableMethods::request::kMethodInfo] = [](Reader& visitor)
        {
                if (visitor.HasMember("error"))
                        return  Rsp_Error::ReflectReader(visitor);

                return java_listOverridableMethods::response::ReflectReader(visitor);
        };
        handler.method2response[java_listOverridableMethods::request::kMethodInfo] = [](Reader& visitor)
        {
                if (visitor.HasMember("error"))
                        return  Rsp_Error::ReflectReader(visitor);

                return java_listOverridableMethods::response::ReflectReader(visitor);
        };

        handler.method2response[java_checkHashCodeEqualsStatus::request::kMethodInfo] = [](Reader& visitor)
        {
                if (visitor.HasMember("error"))
                        return  Rsp_Error::ReflectReader(visitor);

                return java_checkHashCodeEqualsStatus::response::ReflectReader(visitor);
        };


        handler.method2response[java_addOverridableMethods::request::kMethodInfo] = [](Reader& visitor)
        {
                if (visitor.HasMember("error"))
                        return  Rsp_Error::ReflectReader(visitor);

                return java_addOverridableMethods::response::ReflectReader(visitor);
        };

        handler.method2response[java_checkConstructorsStatus::request::kMethodInfo] = [](Reader& visitor)
        {
                if (visitor.HasMember("error"))
                        return  Rsp_Error::ReflectReader(visitor);

                return java_checkConstructorsStatus::response::ReflectReader(visitor);
        };


        handler.method2response[java_checkDelegateMethodsStatus::request::kMethodInfo] = [](Reader& visitor)
        {
                if (visitor.HasMember("error"))
                        return  Rsp_Error::ReflectReader(visitor);

                return java_checkDelegateMethodsStatus::response::ReflectReader(visitor);
        };
        handler.method2response[java_checkToStringStatus::request::kMethodInfo] = [](Reader& visitor)
        {
                if (visitor.HasMember("error"))
                        return  Rsp_Error::ReflectReader(visitor);

                return java_checkToStringStatus::response::ReflectReader(visitor);
        };


        handler.method2response[java_generateAccessors::request::kMethodInfo] = [](Reader& visitor)
        {
                if (visitor.HasMember("error"))
                        return  Rsp_Error::ReflectReader(visitor);

                return java_generateAccessors::response::ReflectReader(visitor);
        };
        handler.method2response[java_generateConstructors::request::kMethodInfo] = [](Reader& visitor)
        {
                if (visitor.HasMember("error"))
                        return  Rsp_Error::ReflectReader(visitor);

                return java_generateConstructors::response::ReflectReader(visitor);
        };
        handler.method2response[java_generateDelegateMethods::request::kMethodInfo] = [](Reader& visitor)
        {
                if (visitor.HasMember("error"))
                        return  Rsp_Error::ReflectReader(visitor);

                return java_generateDelegateMethods::response::ReflectReader(visitor);
        };

        handler.method2response[java_generateHashCodeEquals::request::kMethodInfo] = [](Reader& visitor)
        {
                if (visitor.HasMember("error"))
                        return  Rsp_Error::ReflectReader(visitor);

                return java_generateHashCodeEquals::response::ReflectReader(visitor);
        };
        handler.method2response[java_generateToString::request::kMethodInfo] = [](Reader& visitor)
        {
                if (visitor.HasMember("error"))
                        return  Rsp_Error::ReflectReader(visitor);

                return java_generateToString::response::ReflectReader(visitor);
        };

        handler.method2response[java_generateToString::request::kMethodInfo] = [](Reader& visitor)
        {
                if (visitor.HasMember("error"))
                        return  Rsp_Error::ReflectReader(visitor);

                return java_generateToString::response::ReflectReader(visitor);
        };

        handler.method2response[java_getMoveDestinations::request::kMethodInfo] = [](Reader& visitor)
        {
                if (visitor.HasMember("error"))
                        return  Rsp_Error::ReflectReader(visitor);

                return java_getMoveDestinations::response::ReflectReader(visitor);
        };

        handler.method2response[java_getRefactorEdit::request::kMethodInfo] = [](Reader& visitor)
        {
                if (visitor.HasMember("error"))
                        return  Rsp_Error::ReflectReader(visitor);

                return java_getRefactorEdit::response::ReflectReader(visitor);
        };

        handler.method2response[java_move::request::kMethodInfo] = [](Reader& visitor)
        {
                if (visitor.HasMember("error"))
                        return  Rsp_Error::ReflectReader(visitor);

                return java_move::response ::ReflectReader(visitor);
        };

        handler.method2response[java_organizeImports::request::kMethodInfo] = [](Reader& visitor)
        {
                if (visitor.HasMember("error"))
                        return  Rsp_Error::ReflectReader(visitor);

                return java_organizeImports::response::ReflectReader(visitor);
        };

        handler.method2response[java_resolveUnimplementedAccessors::request::kMethodInfo] = [](Reader& visitor)
        {
                if (visitor.HasMember("error"))
                        return  Rsp_Error::ReflectReader(visitor);

                return java_resolveUnimplementedAccessors::response::ReflectReader(visitor);
        };

        handler.method2response[java_searchSymbols::request::kMethodInfo] = [](Reader& visitor)
        {
                if (visitor.HasMember("error"))
                        return  Rsp_Error::ReflectReader(visitor);

                return java_searchSymbols::response::ReflectReader(visitor);
        };

        handler.method2request[WorkspaceConfiguration::request::kMethodInfo] = [](Reader& visitor)
        {
                return WorkspaceConfiguration::request::ReflectReader(visitor);
        };
        handler.method2request[WorkspaceFolders::request::kMethodInfo] = [](Reader& visitor)
        {
                return WorkspaceFolders::request::ReflectReader(visitor);
        };

}

void AddNotifyJsonRpcMethod(MessageJsonHandler& handler)
{

        handler.method2notification[Notify_Exit::notify::kMethodInfo] = [](Reader& visitor)
        {
                return Notify_Exit::notify::ReflectReader(visitor);
        };
        handler.method2notification[Notify_InitializedNotification::notify::kMethodInfo] = [](Reader& visitor)
        {
                return Notify_InitializedNotification::notify::ReflectReader(visitor);
        };

        handler.method2notification[java_projectConfigurationUpdate::notify::kMethodInfo] = [](Reader& visitor)
        {
                return java_projectConfigurationUpdate::notify::ReflectReader(visitor);
        };

        handler.method2notification[Notify_TextDocumentDidChange::notify::kMethodInfo] = [](Reader& visitor)
        {
                return Notify_TextDocumentDidChange::notify::ReflectReader(visitor);
        };

        handler.method2notification[Notify_TextDocumentDidClose::notify::kMethodInfo] = [](Reader& visitor)
        {
                return Notify_TextDocumentDidClose::notify::ReflectReader(visitor);
        };


        handler.method2notification[Notify_TextDocumentDidOpen::notify::kMethodInfo] = [](Reader& visitor)
        {
                return Notify_TextDocumentDidOpen::notify::ReflectReader(visitor);
        };

        handler.method2notification[Notify_TextDocumentDidSave::notify::kMethodInfo] = [](Reader& visitor)
        {
                return Notify_TextDocumentDidSave::notify::ReflectReader(visitor);
        };

        handler.method2notification[Notify_TextDocumentPublishDiagnostics::notify::kMethodInfo] = [](Reader& visitor)
        {
                return Notify_TextDocumentPublishDiagnostics::notify::ReflectReader(visitor);
        };
        handler.method2notification[Notify_semanticHighlighting::notify::kMethodInfo] = [](Reader& visitor)
        {
                return Notify_semanticHighlighting::notify::ReflectReader(visitor);
        };
        handler.method2notification[td_willSave::notify::kMethodInfo] = [](Reader& visitor)
        {
                return td_willSave::notify::ReflectReader(visitor);
        };

        handler.method2notification[Notify_LogMessage::notify::kMethodInfo] = [](Reader& visitor)
        {
                return Notify_LogMessage::notify::ReflectReader(visitor);
        };
        handler.method2notification[Notify_ShowMessage::notify::kMethodInfo] = [](Reader& visitor)
        {
                return Notify_ShowMessage::notify::ReflectReader(visitor);
        };
        handler.method2notification[Notify_WorkspaceDidChangeWorkspaceFolders::notify::kMethodInfo] = [](Reader& visitor)
        {
                return Notify_WorkspaceDidChangeWorkspaceFolders::notify::ReflectReader(visitor);
        };

        handler.method2notification[Notify_WorkspaceDidChangeConfiguration::notify::kMethodInfo] = [](Reader& visitor)
        {
                return Notify_WorkspaceDidChangeConfiguration::notify::ReflectReader(visitor);
        };


        handler.method2notification[Notify_WorkspaceDidChangeWatchedFiles::notify::kMethodInfo] = [](Reader& visitor)
        {
                return Notify_WorkspaceDidChangeWatchedFiles::notify::ReflectReader(visitor);
        };

        handler.method2notification[Notify_sendNotification::notify::kMethodInfo] = [](Reader& visitor)
        {
                return Notify_sendNotification::notify::ReflectReader(visitor);
        };
        handler.method2notification[lang_status::notify::kMethodInfo] = [](Reader& visitor)
        {
                return lang_status::notify::ReflectReader(visitor);
        };
        handler.method2notification[lang_actionableNotification::notify::kMethodInfo] = [](Reader& visitor)
        {
                return lang_actionableNotification::notify::ReflectReader(visitor);
        };
        handler.method2notification[lang_progressReport::notify::kMethodInfo] = [](Reader& visitor)
        {
                return lang_progressReport::notify::ReflectReader(visitor);
        };
        handler.method2notification[lang_eventNotification::notify::kMethodInfo] = [](Reader& visitor)
        {
                return lang_eventNotification::notify::ReflectReader(visitor);
        };
}

void AddRequstJsonRpcMethod(MessageJsonHandler& handler)
{
        handler.method2request[Req_ClientRegisterCapability::request::kMethodInfo]= [](Reader& visitor)
        {

                return Req_ClientRegisterCapability::request::ReflectReader(visitor);
        };
        handler.method2request[Req_ClientUnregisterCapability::request::kMethodInfo] = [](Reader& visitor)
        {

                return Req_ClientUnregisterCapability::request::ReflectReader(visitor);
        };
}

void AddStandardRequestJsonRpcMethod(MessageJsonHandler& handler)
{

        handler.method2request[td_initialize::request::kMethodInfo] = [](Reader& visitor)
        {

                return td_initialize::request::ReflectReader(visitor);
        };
        handler.method2request[td_shutdown::request::kMethodInfo] = [](Reader& visitor)
        {

                return td_shutdown::request::ReflectReader(visitor);
        };
        handler.method2request[td_codeAction::request::kMethodInfo] = [](Reader& visitor)
        {


                return td_codeAction::request::ReflectReader(visitor);
        };
        handler.method2request[td_codeLens::request::kMethodInfo] = [](Reader& visitor)
        {

                return td_codeLens::request::ReflectReader(visitor);
        };
        handler.method2request[td_completion::request::kMethodInfo] = [](Reader& visitor)
        {

                return td_completion::request::ReflectReader(visitor);
        };

        handler.method2request[td_definition::request::kMethodInfo] = [](Reader& visitor)
        {

                return td_definition::request::ReflectReader(visitor);
        };
        handler.method2request[td_declaration::request::kMethodInfo] = [](Reader& visitor)
        {

                return td_declaration::request::ReflectReader(visitor);
        };
        handler.method2request[td_willSaveWaitUntil::request::kMethodInfo] = [](Reader& visitor)
        {
                if (visitor.HasMember("error"))
                        return  Rsp_Error::ReflectReader(visitor);
                return td_willSaveWaitUntil::request::ReflectReader(visitor);
        };

        handler.method2request[td_highlight::request::kMethodInfo] = [](Reader& visitor)
        {

                return td_highlight::request::ReflectReader(visitor);
        };

        handler.method2request[td_links::request::kMethodInfo] = [](Reader& visitor)
        {

                return td_links::request::ReflectReader(visitor);
        };

        handler.method2request[td_linkResolve::request::kMethodInfo] = [](Reader& visitor)
        {

                return td_linkResolve::request::ReflectReader(visitor);
        };

        handler.method2request[td_symbol::request::kMethodInfo] = [](Reader& visitor)
        {

                return td_symbol::request::ReflectReader(visitor);
        };

        handler.method2request[td_formatting::request::kMethodInfo] = [](Reader& visitor)
        {

                return td_formatting::request::ReflectReader(visitor);
        };

        handler.method2request[td_hover::request::kMethodInfo] = [](Reader& visitor)
        {
                return td_hover::request::ReflectReader(visitor);
        };

        handler.method2request[td_implementation::request::kMethodInfo] = [](Reader& visitor)
        {

                return td_implementation::request::ReflectReader(visitor);
        };

        handler.method2request[td_didRenameFiles::request::kMethodInfo] = [](Reader& visitor)
        {

                return td_didRenameFiles::request::ReflectReader(visitor);
        };

        handler.method2request[td_willRenameFiles::request::kMethodInfo] = [](Reader& visitor)
        {
                return td_willRenameFiles::request::ReflectReader(visitor);
        };
}


lsp::ProtocolJsonHandler::ProtocolJsonHandler()
{
        AddStadardResponseJsonRpcMethod(*this);
        AddJavaExtentionResponseJsonRpcMethod(*this);
        AddNotifyJsonRpcMethod(*this);
        AddStandardRequestJsonRpcMethod(*this);
        AddRequstJsonRpcMethod(*this);
}

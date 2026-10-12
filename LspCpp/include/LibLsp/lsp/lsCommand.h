#pragma once

#include "LibLsp/JsonRpc/serializer.h"

#include <string>
#include <vector>
#include "lsAny.h"
#include "lsMarkedString.h"
//
//Represents a reference to a command.Provides a title which will be used to represent a command in the UI.
//Commands are identified by a string identifier.
//The recommended way to handle commands is to implement their execution on the server side
//if the clientand server provides the corresponding capabilities.Alternatively the tool
//extension code could handle the command.The protocol currently doesn't specify a set of well - known commands.
template<typename AnyArray>
struct lsCommand
{
    // Title of the command (ie, 'save')
    std::string title;
    // Actual command identifier.
    std::string command;
    // Arguments to run the command with.
    // **NOTE** This must be serialized as an array. Use
    // MAKE_REFLECT_STRUCT_WRITER_AS_ARRAY.
    optional<AnyArray> arguments;
    // An optional tooltip shown for this command.
    //
    // @since 3.18.0
    optional<std::pair<optional<std::string>, optional<MarkupContent>>> tooltip;

    void swap(lsCommand<AnyArray>& arg) noexcept
    {
        title.swap(arg.title);
        command.swap(arg.command);
        arguments.swap(arg.arguments);
        tooltip.swap(arg.tooltip);
    }
};
template<typename TVisitor, typename T>
void Reflect(TVisitor& visitor, lsCommand<T>& value)
{
    REFLECT_MEMBER_START();
    REFLECT_MEMBER(title);
    REFLECT_MEMBER(command);
    REFLECT_MEMBER(arguments);
    REFLECT_MEMBER(tooltip);
    REFLECT_MEMBER_END();
}

using lsCommandWithAny = lsCommand<std::vector<lsp::Any>>;

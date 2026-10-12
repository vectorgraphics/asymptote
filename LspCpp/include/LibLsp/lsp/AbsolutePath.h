#pragma once

#include <LibLsp/JsonRpc/serializer.h>
#include <iosfwd>
#include <string>

struct AbsolutePath
{
    AbsolutePath();

    static AbsolutePath FromNormalized(std::string const& absolute_path);

    explicit AbsolutePath(std::string const& path);

    std::string const& path() const;
    bool valid() const;
    bool empty() const;
    bool is_absolute() const;

    bool operator==(AbsolutePath const& rhs) const;
    bool operator!=(AbsolutePath const& rhs) const;
    bool operator<(AbsolutePath const& rhs) const;
    bool operator>(AbsolutePath const& rhs) const;

private:
    std::string path_;
};

void Reflect(Reader& visitor, AbsolutePath& value);
void Reflect(Writer& visitor, AbsolutePath& value);
std::ostream& operator<<(std::ostream& out, AbsolutePath const& path);

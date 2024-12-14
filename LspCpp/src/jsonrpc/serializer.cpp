#include "LibLsp/JsonRpc/serializer.h"
#include <stdexcept>
#include <rapidjson/allocators.h>
#include "LibLsp/JsonRpc/json.h"



//// Elementary types

void JsonNull::swap(JsonNull& arg) noexcept
{
}


void Reflect(Reader& visitor, uint8_t& value) {
  if (!visitor.IsInt())
    throw std::invalid_argument("uint8_t");
  value = (uint8_t)visitor.GetInt();
}
void Reflect(Writer& visitor, uint8_t& value) {
  visitor.Int(value);
}

void Reflect(Reader& visitor, short& value) {
  if (!visitor.IsInt())
    throw std::invalid_argument("short");
  value = (short)visitor.GetInt();
}
void Reflect(Writer& visitor, short& value) {
  visitor.Int(value);
}

void Reflect(Reader& visitor, unsigned short& value) {
  if (!visitor.IsInt())
    throw std::invalid_argument("unsigned short");
  value = (unsigned short)visitor.GetInt();
}
void Reflect(Writer& visitor, unsigned short& value) {
  visitor.Int(value);
}

void Reflect(Reader& visitor, int& value) {
  if (!visitor.IsInt())
    throw std::invalid_argument("int");
  value = visitor.GetInt();
}
void Reflect(Writer& visitor, int& value) {
  visitor.Int(value);
}

void Reflect(Reader& visitor, unsigned& value) {
  if (!visitor.IsUint64())
    throw std::invalid_argument("unsigned");
  value = visitor.GetUint32();
}
void Reflect(Writer& visitor, unsigned& value) {
  visitor.Uint32(value);
}

void Reflect(Reader& visitor, long& value) {
  if (!visitor.IsInt64())
    throw std::invalid_argument("long");
  value = long(visitor.GetInt64());
}
void Reflect(Writer& visitor, long& value) {
  visitor.Int64(value);
}

void Reflect(Reader& visitor, unsigned long& value) {
  if (!visitor.IsUint64())
    throw std::invalid_argument("unsigned long");
  value = (unsigned long)visitor.GetUint64();
}
void Reflect(Writer& visitor, unsigned long& value) {
  visitor.Uint64(value);
}

void Reflect(Reader& visitor, long long& value) {
  if (!visitor.IsInt64())
    throw std::invalid_argument("long long");
  value = visitor.GetInt64();
}
void Reflect(Writer& visitor, long long& value) {
  visitor.Int64(value);
}

void Reflect(Reader& visitor, unsigned long long& value) {
  if (!visitor.IsUint64())
    throw std::invalid_argument("unsigned long long");
  value = visitor.GetUint64();
}
void Reflect(Writer& visitor, unsigned long long& value) {
  visitor.Uint64(value);
}

void Reflect(Reader& visitor, double& value) {
  if (!visitor.IsNumber())
    throw std::invalid_argument("double");
  value = visitor.GetDouble();
}
void Reflect(Writer& visitor, double& value) {
  visitor.Double(value);
}

void Reflect(Reader& visitor, bool& value) {
  if (!visitor.IsBool())
    throw std::invalid_argument("bool");
  value = visitor.GetBool();
}
void Reflect(Writer& visitor, bool& value) {
  visitor.Bool(value);
}

void Reflect(Reader& visitor, std::string& value) {
  if (!visitor.IsString())
    throw std::invalid_argument("std::string");
  value = visitor.GetString();
}
void Reflect(Writer& visitor, std::string& value) {
  visitor.String(value.c_str(), (rapidjson::SizeType)value.size());
}

void Reflect(Reader& visitor, JsonNull& value) {
  visitor.GetNull();
}

void Reflect(Writer& visitor, JsonNull& value) {
  visitor.Null();
}


void Reflect(Reader& visitor, SerializeFormat& value) {
  std::string fmt = visitor.GetString();
  value = fmt[0] == 'm' ? SerializeFormat::MessagePack : SerializeFormat::Json;
}

void Reflect(Writer& visitor, SerializeFormat& value) {
  switch (value) {
    case SerializeFormat::Json:
      visitor.String("json");
      break;
    case SerializeFormat::MessagePack:
      visitor.String("msgpack");
      break;
  }
}


std::string JsonReader::ToString() const
{
        rapidjson::StringBuffer strBuf;
        strBuf.Clear();
        rapidjson::Writer<rapidjson::StringBuffer> writer(strBuf);
        m_->Accept(writer);
        std::string strJson = strBuf.GetString();
        return strJson;
}

void JsonReader::IterMap(std::function<void(const char*, Reader&)> fn)
{
        path_.push_back("0");
        for (auto& entry : m_->GetObject())
        {
                auto saved = m_;
                m_ = &(entry.value);

                fn(entry.name.GetString(), *this);
                m_ = saved;
        }
        path_.pop_back();
}

 void JsonReader::IterArray(std::function<void(Reader&)> fn)
{
        if (!m_->IsArray())
                throw std::invalid_argument("array");
        // Use "0" to indicate any element for now.
        path_.push_back("0");
        for (auto& entry : m_->GetArray())
        {
                auto saved = m_;
                m_ = &entry;
                fn(*this);
                m_ = saved;
        }
        path_.pop_back();
}

void JsonReader::DoMember(const char* name, std::function<void(Reader&)> fn)
{
        path_.push_back(name);
        auto it = m_->FindMember(name);
        if (it != m_->MemberEnd())
        {
                auto saved = m_;
                m_ = &it->value;
                fn(*this);
                m_ = saved;
        }
        path_.pop_back();
}

std::string JsonReader::GetPath() const
{
        std::string ret;
        for (auto& t : path_)
        {
                ret += '/';
                ret += t;
        }
        ret.pop_back();
        return ret;
}


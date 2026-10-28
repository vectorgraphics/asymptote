//
// Created by jamie on 8/23/21.
//

#include "EXRFiles.h"
#include <tinyexr.h>
#include "locate.h"
#include "settings.h"
#include "camperror.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <cstdio>

#ifdef HAVE_LIBCURL
#include <curl/curl.h>
#endif



namespace camp
{
using std::cout;
using std::cerr;
using std::endl;

#ifdef HAVE_LIBCURL

// Write callback for curl: writes to the given ofstream and counts bytes.
struct CurlDownload {
  std::ofstream* out;
  long long bytes = 0;
};
static size_t curlWriteCb(char* data, size_t size, size_t nmemb, void* ud)
{
  auto* ctx = static_cast<CurlDownload*>(ud);
  ctx->out->write(data, size * nmemb);
  if (ctx->out->good()) {
    ctx->bytes += size * nmemb;
    return size * nmemb;
  }
  return 0;
}
// Human-readable byte count (e.g. "5.2 MB").
static string humanBytes(long long n)
{
  char buf[32];
  if (n >= 1048576)
    snprintf(buf, sizeof(buf), "%.1f MB", (double)n / 1048576);
  else if (n >= 1024)
    snprintf(buf, sizeof(buf), "%.1f KB", (double)n / 1024);
  else
    snprintf(buf, sizeof(buf), "%lld B", n);
  return buf;
}

// Download url to localFile, returning true on success.
// Writes to a .dl temp file first so a failure doesn't leave a partial EXR.
static bool downloadFile(string const& url, string const& localFile)
{
  string tmpFile = localFile + ".dl";

  std::ofstream out(tmpFile.c_str(), std::ios::binary);
  if (!out.good())
    return false;

  CurlDownload ctx{&out};

  CURL* curl = curl_easy_init();
  if (!curl) {
    out.close();
    return false;
  }
  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
  curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 1L);
  curl_easy_setopt(curl, CURLOPT_FAILONERROR, 1L);
  curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 10L);
  curl_easy_setopt(curl, CURLOPT_TIMEOUT, 300L);
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curlWriteCb);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &ctx);

  CURLcode res = curl_easy_perform(curl);
  long httpCode = 0;
  curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &httpCode);
  curl_easy_cleanup(curl);
  out.close();

  if (res != CURLE_OK || httpCode != 200) {
    std::filesystem::remove(tmpFile.c_str());
    char codebuf[16];
    if (httpCode)
      snprintf(codebuf, sizeof(codebuf), "HTTP %ld", httpCode);
    else
      snprintf(codebuf, sizeof(codebuf), "%s", curl_easy_strerror(res));
    cerr << "  " << codebuf << " (" << url << ")" << endl;
    return false;
  }
  std::error_code ec;
  std::filesystem::rename(tmpFile.c_str(), localFile.c_str(), ec);
  if (ec) {
    std::filesystem::remove(tmpFile.c_str());
    return false;
  }
  cerr << "    " << humanBytes(ctx.bytes) << endl;
  return true;
}

#endif // HAVE_LIBCURL

// The user-writable IBL base directory: <initdir>/ibl/
// (i.e. ~/.asy/ibl/ by default). Since initdir is in the search path,
// locateFile("ibl") will find it once it exists.
static string iblBaseDir()
{
  if (settings::initdir.empty())
    return "";
  string base = settings::initdir + "/ibl";
  std::error_code ec;
  std::filesystem::create_directories(base.c_str(), ec);
  return base + "/";
}

#ifdef HAVE_LIBCURL

// Download the IBL environment map `image` into iblBase. Returns true on
// success. Downloads automatically (no prompt) -- the user opted in by
// requesting the image via -image. Set -image="" to disable.
// `image` is meant to be a file name, possibly with a relative
// subdirectory of the image library; it must resolve within the library
// directory. settings.image and settings.imageURL are settable from .asy
// code at any time, so the restrictions below are checked on every call:
// in safe mode we only auto-download from the default URL (a malicious
// .asy file could otherwise overload settings.imageURL to track users or
// serve arbitrary content); '%' is rejected outright since it is
// ambiguous in the URL; and ".." components are rejected so the name
// cannot resolve outside the library directory.
static bool downloadIBL(string const& image, string const& iblBase)
{
  string const url = settings::getSetting<string>("imageURL");
  // Only auto-download from the default URL unless -nosafe is specified.
  if (url != settings::defaultImageURL && settings::safe) {
    cerr << "Not auto-downloading IBL image '" << image
         << "' from non-default -imageURL. Use -nosafe to allow this."
         << endl;
    return false;
  }

  if (!settings::iblNameOK(image)) {
    cerr << "IBL image name '" << image
         << "' is not allowed for automatic download." << endl;
    return false;
  }
  string imgDir = iblBase + image + "/";

  // Count how many files are still missing.
  int missing = 0;
  if (!std::filesystem::exists((iblBase + "refl.exr").c_str()))
    ++missing;
  if (!std::filesystem::exists((imgDir + "diffuse.exr").c_str()))
    ++missing;
  for (int i = 0; i <= 10; ++i) {
    char fname[64];
    snprintf(fname, sizeof(fname), "refl%d.exr", i);
    if (!std::filesystem::exists((imgDir + fname).c_str()))
      ++missing;
  }

  // Nothing to do.
  if (missing == 0)
    return true;

  cerr << "Downloading IBL environment map '" << image << "' ("
       << missing << " file" << (missing == 1 ? "" : "s") << ") to "
       << iblBase << image << "..." << endl;

  std::error_code ec;
  std::filesystem::create_directories(imgDir.c_str(), ec);

  int downloaded = 0, skipped = 0, failed = 0;

  // Shared refl.exr
  {
    string local = iblBase + "refl.exr";
    if (std::filesystem::exists(local.c_str())) {
      ++skipped;
    } else {
      cerr << "  refl.exr" << endl;
      string remote = url + "/refl.exr";
      if (downloadFile(remote, local)) ++downloaded;
      else ++failed;
    }
  }

  // diffuse.exr -- if this fails the image doesn't exist; abort.
  {
    string local = imgDir + "diffuse.exr";
    if (std::filesystem::exists(local.c_str())) {
      ++skipped;
    } else {
      cerr << "  " << image << "/diffuse.exr" << endl;
      string remote = url + "/" + image + "/diffuse.exr";
      if (!downloadFile(remote, local)) {
        cerr << "IBL environment map '" << image << "' not found on server." << endl;
        // remove imgDir only if empty (keeps partial downloads for resume)
        std::filesystem::remove(imgDir.c_str(), ec);
        return false;
      }
      ++downloaded;
    }
  }

  // refl0.exr .. refl10.exr
  for (int i = 0; i <= 10; ++i) {
    char fname[64];
    snprintf(fname, sizeof(fname), "refl%d.exr", i);
    string name(fname);
    string local = imgDir + name;
    if (std::filesystem::exists(local.c_str())) {
      ++skipped;
    } else {
      cerr << "  " << image << "/" << name << endl;
      string remote = url + "/" + image + "/" + name;
      if (downloadFile(remote, local)) ++downloaded;
      else ++failed;
    }
  }

  if (failed > 0) {
    cerr << "Download incomplete: " << failed << " file(s) failed." << endl;
    // remove imgDir only if empty (keeps partial downloads for resume)
    std::filesystem::remove(imgDir.c_str(), ec);
    return false;
  }
  cerr << "IBL environment map '" << image << "' ready ("
       << downloaded << " downloaded, " << skipped << " already present)." << endl;
  return true;
}

#endif // HAVE_LIBCURL

// Ensure the IBL image library is available. Returns the resolved directory
// path (with trailing slash) or "" if unavailable.
string ensureIBLDir()
{
  string const image = settings::getSetting<string>("image");
  if (image.empty())
    return "";
  string dir = settings::locateFile(settings::getSetting<string>("imageDir"));

  // Determine the download target: use the ibl directory if it exists,
  // otherwise fall back to ~/.asy/ibl/.
  string iblBase;
  if (!dir.empty())
    iblBase = dir + "/";
  else
    iblBase = iblBaseDir();
  if (iblBase.empty())
    return "";

  // downloadIBL skips files that already exist, so it is safe to call
  // unconditionally: a no-op if everything is in place, and fills in any
  // missing files otherwise (interrupted downloads, clobbered files, etc.).
  // Returns false if files are still missing (download failed or declined).
  bool available;
#ifdef HAVE_LIBCURL
  available = downloadIBL(image, iblBase);
  if (available) {
    string newDir = settings::locateFile(settings::getSetting<string>("imageDir"));
    if (!newDir.empty())
      dir = newDir;
    else
      dir = iblBase;
  }
#else
  available = !dir.empty();
  if (!available) {
    cerr << "EXR file not found. Precomputed image directories like ibl/"
         << image << " can be downloaded into the Asymptote search path "
         << "like this:" << endl << endl
         << "wget -q --show-progress -nH -np -r --cut-dirs=1 "
         << settings::getSetting<string>("imageURL") << "/refl.exr" << endl
         << "wget -q --show-progress -nH -np -r --cut-dirs=1 "
         << settings::getSetting<string>("imageURL") << "/" << image
         << endl;
  }
#endif
  if (!available)
    return "";
  if (settings::verbose > 1)
    cerr << "Reading IBL images from " << dir + "/" + image << endl;
  return dir + "/";
}

IEXRFile::IEXRFile(const string& File)
{
  char const* err;
  int ret;
  string file=settings::locateFile(File);
  string image=settings::getSetting<string>("image");
  if(file.empty()) {
    ostringstream os;
    os << "EXR file not found: " << File
       << endl << "IBL environment map '" << image << "' may need to be downloaded.";
    reportError(os);
  }
  const char *filename=file.c_str();
  if((ret=LoadEXR(&data,&width,&height,filename,&err)) != TINYEXR_SUCCESS)
    {
      ostringstream os;
      os << "TinyEXR Error: " << err << " " << filename;
      FreeEXRErrorMessage(err);
      reportError(os);
    }
}

IEXRFile::~IEXRFile()
{
  if(data) {
    free(data);
    data=nullptr;
  }
}

}

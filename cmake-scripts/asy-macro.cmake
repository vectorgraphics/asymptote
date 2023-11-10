list(APPEND ASY_MACROS
        PACKAGE_NAME="${ASY_PACKAGE_NAME}"
        PACKAGE_VERSION="${ASY_VERSION}"
        PACKAGE_BUGREPORT="${ASY_BUGREPORT}"
)

# Since we require C++11 and up, some macros are automatically included
list(APPEND ASY_MACROS
        HAVE_UNORDERED_MAP
        HAVE_STRFTIME
)

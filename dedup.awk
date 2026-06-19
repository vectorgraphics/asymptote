{
  n = split($0, w, " ")
  i = 1
  while (i <= n) {
    if (w[i] ~ /^-/ && i < n && w[i+1] !~ /^-/) {
      u = w[i] " " w[i+1]
      i += 2
    } else {
      u = w[i]
      i++
    }
    if (!(u in s)) {
      s[u] = 1
      o[++c] = u
    }
  }
  for (i = 1; i <= c; i++) {
    if (i > 1) printf " "
    printf "%s", o[i]
  }
  print ""
}

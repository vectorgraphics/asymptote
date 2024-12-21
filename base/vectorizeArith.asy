typedef import(Number);

bool[] operator ==(Number[] r, Number s)
{
  return sequence(new bool(int i) {return r[i] == s;},r.length);
}

bool operator ==(Number[] r, Number[] s)
{
  if(r.length != s.length)
    abort(" operation attempted on arrays of different lengths: "+
          string(r.length)+" != "+string(s.length));
  return all(sequence(new bool(int i) {return r[i] == s[i];},r.length));
}

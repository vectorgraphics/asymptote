access version;
if(version.VERSION != VERSION) {
  warning("version","using possibly incompatible version "+
          version.VERSION+" of plain.asy"+'\n');
  nowarn("version");
}

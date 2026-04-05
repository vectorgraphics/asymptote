// Autounravel modifier errors.
{
  struct A {
    static static int x;  // too many static modifiers
    autounravel static int y;  // no error
    static autounravel int z;  // no error
    autounravel autounravel int w;  // too many autounravel modifiers
    autounravel static autounravel int v;  // too many autounravel modifiers
    static autounravel static int u;  // too many static modifiers
    autounravel struct B {}  // types cannot be autounraveled
  }
}
{
  autounravel int x;  // top-level fields cannot be autounraveled
}
{
  struct A {
    autounravel int qaz;
    autounravel int qaz;  // cannot shadow autounravel qaz
  }
}
{
  // Even if the first (implicitly defined) instance of a function is allowed
  // to be shadowed, the (explicit) shadower cannot itself be shadowed.
  struct A {
    autounravel bool alias(A, A);  // no error
    autounravel bool alias(A, A);  // cannot shadow autounravel alias
  }
}

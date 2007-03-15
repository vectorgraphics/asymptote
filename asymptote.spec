%{!?_texmf: %define _texmf %(eval "echo `kpsewhich -expand-var '$TEXMFMAIN'`")}

Name:           asymptote
Version:        1.24
Release:        1%{?dist}
Summary:        Descriptive vector graphics language

Group:          Applications/Publishing
License:        GPL
URL:            http://asymptote.sourceforge.net/
Source:         http://dl.sourceforge.net/sourceforge/asymptote/asymptote-%{version}.tar.gz
BuildRoot:      %{_tmppath}/%{name}-%{version}-%{release}-root-%(%{__id_u} -n)

BuildRequires:  ncurses-devel
BuildRequires:  readline-devel
BuildRequires:  fftw-devel >= 3.0
BuildRequires:  gc-devel >= 6.7
BuildRequires:  gsl-devel
BuildRequires:  tetex-latex
BuildRequires:  ghostscript
BuildRequires:  texinfo >= 4.7
BuildRequires:  ImageMagick

Requires:       tetex-latex
Requires:       tkinter
Requires(post): /usr/bin/texhash /sbin/install-info
Requires(postun): /usr/bin/texhash /sbin/install-info

%define texpkgdir   %{_texmf}/tex/latex/%{name}

%description
Asymptote is a powerful descriptive vector graphics language for technical
drawings, inspired by MetaPost but with an improved C++-like syntax.
Asymptote provides for figures the same high-quality level of typesetting
that LaTeX does for scientific text.


%prep
%setup -q
%{__sed} -i 's|^#!/usr/bin/env python$|#!%{__python}|' xasy


%build
CFLAGS="`echo $RPM_OPT_FLAGS | sed s/-O2/-O3/`" \
%configure --enable-gc=system
make %{?_smp_mflags}


%install
rm -rf $RPM_BUILD_ROOT
make install-all DESTDIR=$RPM_BUILD_ROOT

%{__install} -p -m 644 BUGS ChangeLog LICENSE README ReleaseNotes TODO \
    $RPM_BUILD_ROOT%{_defaultdocdir}/%{name}/


%clean
rm -rf $RPM_BUILD_ROOT


%post
texhash >/dev/null 2>&1 || :
/sbin/install-info %{_infodir}/%{name}.info.gz %{_infodir}/dir 2>/dev/null || :

%postun
texhash >/dev/null 2>&1 || :
if [ $1 = 0 ]; then
    /sbin/install-info --remove %{_infodir}/%{name}.info.gz %{_infodir}/dir 2>/dev/null || :
fi


%files
%defattr(-,root,root,-)
%doc %{_defaultdocdir}/%{name}/
%{_bindir}/*
%{_datadir}/%{name}/
%{texpkgdir}/
%{_mandir}/man1/*.1*
%{_infodir}/*.info*


%changelog
* Fri May 30 2006 John Bowman <> - 1.07-1
- Use make install-all to also install info pages.

* Fri May 26 2006 Jose Pedro Oliveira <jpo at di.uminho.pt> - 1.07-1
- Update to 1.07.

* Sun May 21 2006 John Bowman <> - 1.06-1
- Update to 1.06.

* Mon May  8 2006 John Bowman <> - 1.05-1
- Update to 1.05.

* Sun May  7 2006 Jose Pedro Oliveira <jpo at di.uminho.pt> - 1.04-1
- Update to 1.04.

* Fri Mar 31 2006 Jose Pedro Oliveira <jpo at di.uminho.pt> - 1.03-1
- Update to 1.03.

* Thu Mar 23 2006 Jose Pedro Oliveira <jpo at di.uminho.pt> - 1.02-1
- First build.

%{!?_texmf: %global _texmf %(eval "echo `kpsewhich -expand-var '$TEXMFLOCAL'`")}
%global _python_bytecompile_errors_terminate_build 0
%global __python %{__python3}

Name:           asymptote
Version:        2.65
Release:        1%{?dist}
Summary:        Descriptive vector graphics language

Group:          Applications/Publishing
License:        GPL
URL:            http://asymptote.sourceforge.net/
Source:         http://downloads.sourceforge.net/sourceforge/asymptote/asymptote-%{version}.src.tgz
BuildRoot:      %{_tmppath}/%{name}-%{version}-%{release}-root-%(%{__id_u} -n)

BuildRequires:  ncurses-devel
BuildRequires:  readline-devel
BuildRequires:  fftw-devel >= 3.0
BuildRequires:  gc-devel >= 6.7
BuildRequires:  gsl-devel
BuildRequires:  glm-devel
BuildRequires:  tetex-latex
BuildRequires:  ghostscript >= 9.14
BuildRequires:  texinfo >= 4.7
BuildRequires:  ImageMagick

Requires:       tetex-latex
Requires:       freeglut-devel >= 3.0.0
Requires(post): /usr/bin/texhash /sbin/install-info
Requires(postun): /usr/bin/texhash /sbin/install-info

%description
Asymptote is a powerful descriptive vector graphics language for technical
drawings, inspired by MetaPost but with an improved C++-like syntax.
Asymptote provides for figures the same high-quality level of typesetting
that LaTeX does for scientific text.


%prep
%setup -q


%build
CFLAGS="`echo $RPM_OPT_FLAGS | sed s/-O2/-O3/`" \
%configure --with-latex=%{_texmf}/tex/latex --with-context=%{_texmf}/tex/context/third
make %{?_smp_mflags}


%install
rm -rf $RPM_BUILD_ROOT
make install-notexhash DESTDIR=$RPM_BUILD_ROOT

%{__install} -p -m 644 BUGS ChangeLog LICENSE README ReleaseNotes TODO \
    $RPM_BUILD_ROOT%{_defaultdocdir}/%{name}/


%clean
rm -rf $RPM_BUILD_ROOT


%post
texhash >/dev/null 2>&1 || :
/sbin/install-info %{_infodir}/%{name}/%{name}.info.gz %{_infodir}/dir 2>/dev/null || :

%postun
texhash >/dev/null 2>&1 || :
if [ $1 = 0 ]; then
    /sbin/install-info --remove %{_infodir}/%{name}/%{name}.info.gz %{_infodir}/dir 2>/dev/null || :
fi


%files
%defattr(-,root,root,-)
%doc %{_defaultdocdir}/%{name}/
%{_bindir}/*
%{_datadir}/%{name}/
%{_texmf}/tex/latex/%{name}
%{_texmf}/tex/context/third/%{name}
%{_mandir}/man1/*.1*
%{_infodir}/%{name}/
%{_infodir}/%{name}/*.info*
%{_infodir}/*.info*


%changelog
* Thu Apr 19 2007 John Bowman <> - 1.26-1
- Update source tar ball name.

* Tue May 30 2006 John Bowman <> - 1.07-1
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

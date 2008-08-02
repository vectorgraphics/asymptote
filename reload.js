// Reload the document associated with a given path.

// UNIX: Copy to ~/.adobe/Acrobat/x.x/JavaScripts/
// MSWindows: Copy to %APPDATA%/Adobe/Acrobat/x.x/JavaScripts/
// where x.x represents the appropriate Acrobat Reader version number

reload = app.trustedFunction(function(path) {
	app.beginPriv();
	n=app.activeDocs.length;
	for(i=app.activeDocs.length-1; i >= 0; --i) {
	    Doc=app.activeDocs[i];
	    if(Doc.path == path && Doc != this) {
		Doc.closeDoc();
		app.openDoc(path);
		break;
	    }
	}
	app.endPriv();
    });

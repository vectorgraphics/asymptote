// Print without querying user.

// UNIX: Copy to ~/.adobe/Acrobat/x.x/JavaScripts/

// MSWindows: Copy to %APPDATA%/Adobe/Acrobat/x.x/JavaScripts/

// Note: x.x represents the appropriate Acrobat Reader version number.

silentPrint = app.trustedFunction(function(pp) {
	app.beginPriv();
	this.print(pp);
	app.endPriv();
    });

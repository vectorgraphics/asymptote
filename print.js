printToFile = app.trustedFunction(function(fileName) {
	app.beginPriv();
	console.println("Rasterizing to "+fileName);
	var pp = this.getPrintParams();
        pp.interactive = pp.constants.interactionLevel.silent;
	pp.fileName = fileName;
	pp.bitmapDPI = 9600;
	pp.gradientDPI = 9600;
	fv = pp.constants.flagValues;
	// do not auto-rotate
	pp.flags |= fv.suppressRotate;
	// do not scale the page
	pp.pageHandling = pp.constants.handling.none;
	pp.printerName = "FILE";
	this.print(pp);
	app.endPriv();
    });

string latexCodeforPrintingTheLPP(real[] c, real[][] A, real[] b){	
	string cx = '\\mathrm{minimize}\t';
	bool allPreviousEntriesWereZero = true;
	if (c[0] == 0) {
		cx += "&    ";
		allPreviousEntriesWereZero = true;
	} else {
		cx += "&" + ((c[0] != 1) ? string(c[0]) : " ") + "x_1";
		allPreviousEntriesWereZero = false;
	}
	for(int i = 1; i < c.length; ++i) {
		if (c[i] == 0) {
			cx += "&  &  ";
		} else {
			cx += '\t&';
			if (allPreviousEntriesWereZero == false) {
				cx += (c[i] >= 0) ? '+' : '-';
			}
			allPreviousEntriesWereZero = false;
			cx += ' &' + ((c[i] != 1) ? string(abs(c[i])) : " ") + "x_" + string(i+1);
		}
	}
	cx += " && \\ " + '\n';

	string AxEqualsb = "" ;
	bool allPreviousEntriesWereZero = true;
	for(int i = 0; i < b.length; ++i) {
		AxEqualsb += (i == 0) ? '\mathrm{subject\ to} \t' : '\t\t\t\t';	
		if (A[i][0] == 0) {
			AxEqualsb += "&    ";
			allPreviousEntriesWereZero = true;
		} else {
			AxEqualsb += "&" + ((A[i][0] != 1) ? string(A[i][0]) : " ") + "x_1";
			allPreviousEntriesWereZero = false;
		}
		for(int j = 1; j < c.length; ++j) {
			if (A[i][j] == 0) {
				AxEqualsb += '\t&  &    ';
			} else {
				AxEqualsb += '\t&';
				if (allPreviousEntriesWereZero == false) {
					AxEqualsb += (A[i][j] >= 0) ? '+' : '-';
				}
				allPreviousEntriesWereZero = false;
				AxEqualsb += ' &' + ((A[i][j] != 1) ? string(abs(A[i][j])) : " ") + "x_" + string(j+1);
			}

		}
		AxEqualsb += " &= &" + string(b[i]) + " \\ " + '\n';
	}


	string theProblem = "\begin{equation*}" + '\n' + "\begin{array}{l "; 
	for (int i = 0; i < c.length; ++i) {
		theProblem += "r c ";
	}
	theProblem += "r}" + '\n';
	theProblem += cx + AxEqualsb;
	theProblem += '\t\t\t\t&' + '\multicolumn{' + string(c.length+1) + "}{l}{";
	string elementOfCWithMaximumLength = "";
	for (int i = 0; i < b.length; ++i) {
		if (length(string(A[i][1])) > length(elementOfCWithMaximumLength)) {
			elementOfCWithMaximumLength = string(A[i][1]);
		}
	}
	theProblem += "\textcolor{white}{" + elementOfCWithMaximumLength + "}";
	for (int i = 1; i <= c.length ; ++i) {
		theProblem += (i == 1 ? "" : ", ") + "x_" + string(i);	
	}
	theProblem += "\geq 0.}" + '\n\end{array}\n\end{equation*}';	
	return theProblem;
}

string solveLPPUsingTheSimplexMethod(real[] c, real[][] A, real[] b) {	
	string finalText = 'The linear programming problem is \n\n';
	// print the linear programming problem
	finalText += latexCodeforPrintingTheLPP(c, A, b) + '\n'; 

	// ----- Phase I.
	finalText += "Using the two-phase simplex method, the problem is solved as follows.\\" + '\n\n' + "\noindent\textbf{Phase I.}\\" +  '\n\n'; 

	// -- Step I. We check if b >= 0.
	finalText += "\textbf{Step 1.} Every constraint with a negative corresponding entry of $\mathbf{b}$ is multiplied by $-1$ to have $\mathbf{b}\geq\mathbf{0}$";
	bool negative_bEntry = false;
	for (int i = 0; i < b.length; ++i) {
		if (b[i] < 0) {
			b[i] = -b[i];
			A[i] = -A[i];
			negative_bEntry = true;
		} 
	}
	if (negative_bEntry == true) {
		finalText += ":" + '\n' + latexCodeforPrintingTheLPP(c, A, b) + '\n\n';
	} else {
		finalText += ". In this case, there is no need to change the constraints." + '\n\n';
	}

	// -- Step II.
	finalText += "\textbf{Step 2.}";
        real[] c2=concat(array(c.length,0),array(b.length,1));
	real[][] A2 = A; 
        real[][] I=identity(b.length);
	for (int i = 0; i < b.length; ++i) {
          A2[i].append(I[i]);
          //          A2[i].append(sequence(new real(int j) {return j == i ? 1 : 0;},b.length));
	}
	finalText += latexCodeforPrintingTheLPP(c2, A2, b);

	return finalText; 
}

file fout = output("out.tex");
write(fout,"\documentclass{article}
\usepackage{amsmath,color}
\begin{document}
");

real[] c = {1,2,3};
real[][] A = {{2,0,3},{3,2,7}};
real[] b = {5,6};
write(fout,solveLPPUsingTheSimplexMethod(c,A,b));  
write(fout,"\end{document}");

if(settings.verbose > 0) write("Wrote out.tex");


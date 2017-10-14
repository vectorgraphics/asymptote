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

string latexCodeForPrintingTheSimplexTableau(real[] zerothRow, real[][] simplexTableau, real[] firstColumn, real currentCost, real[] basicIndices){
	string theTableau = "\begin{equation*}" + '\n' + "\begin{array}{c l |r|"; 
	for (int i = 0; i < zerothRow.length; ++i) {
		theTableau += "r ";
	}
	theTableau += "|}" + '\n' + "\cline{3-" + string(zerothRow.length + 3) + "}" + '\n';
	theTableau += "& &";
	for (int i = 0; i < zerothRow.length; ++i) {
		theTableau += " &x_" + string(i + 1);
	}
	theTableau += "\\" + '\n' + "\cline{3-" + string(zerothRow.length + 3) + "}" + '\n';
	theTableau += "&  &" + string(currentCost,3); 
	for (int i = 0; i < zerothRow.length; ++i) {
		theTableau += " &" + string(zerothRow[i],2);
	}  
	theTableau += "\\" + '\n' + "\cline{3-" + string(zerothRow.length + 3) + "}" + '\n';
	for (int i = 0; i < basicIndices.length; ++i) {
		theTableau += "x_" + string(basicIndices[i],2) + " &="; 
		theTableau += " &" + string(firstColumn[i],2);
		for (int j = 0; j < zerothRow.length; ++j) {
			theTableau += " &" + string(simplexTableau[i][j],2);
		}
		theTableau += "\\" + '\n'; 
	}
	theTableau += "\cline{3-" + string(zerothRow.length + 3) + "}" + '\n\end{array}\n\end{equation*}';	
	return theTableau;
}

string latexCodeForPrintingSimplexTableaus(real[] c, real[][] A, real[] b, real[] basicIndices) {
	real currentCost = 0;
	string finalText = latexCodeForPrintingTheSimplexTableau(c, A, b, 0, basicIndices);
	while (true) {
		int j = -1;
		for (int i = 0; i < c.length; ++i) {
			if (c[i] < 0) {
				j = i;
				break;
			}
		}
		if (j == -1) {
			finalText += "The current solution is optimal";
			return finalText;
		}
		int[] indexOfPositiveElementsOfColumnJ;	
		int LexicoGraphicallySmallestRow = -1;
		for (int i = 0; i < b.length; ++i) {
			if (A[i][j] > 0) {
				indexOfPositiveElementsOfColumnJ.push(i);
				temp = (b[]) ? i : ;
				for (int k = 0; k < c.length; ++k) {
					A[i][k]
				}
			}
		}
		if (indexOfPositiveElementsOfColumnJ.length == 0) {
			finalText += "The optimal cost is $-\infty$.";
			return finalText;
		} 
		// l is the index of the row that corresponds to the smallest ratio x_{B(i)}/u_i
		int l = indexOfPositiveElementsOfColumnJ[0];	
		real smallestRatioOfXBiToUi = b[indexOfPositiveElementsOfColumnJ[0]]/A[indexOfPositiveElementsOfColumnJ[0]][j] ;
		for (int i = 1; i < indexOfPositiveElementsOfColumnJ.length; ++i) {
			if (b[indexOfPositiveElementsOfColumnJ[i]]/A[indexOfPositiveElementsOfColumnJ[i]][j] < smallestRatioOfXBiToUi) {
				smallestRatioOfXBiToUi = b[indexOfPositiveElementsOfColumnJ[i]]/A[indexOfPositiveElementsOfColumnJ[i]][j];
				l = indexOfPositiveElementsOfColumnJ[i];
				break;
			}
		}
		//column A_B(l) exists and column A_j enters
		basicIndices[l] = j + 1;	
		currentCost = currentCost - b[l]*c[j]/A[l][j];
		real cJ = c[j];
		for (int i = 0; i < c.length; ++i) {
			c[i] -=  A[l][i]*cJ/A[l][j];
		}
		real bL = b[l];
		for(int k = 0; k < b.length; ++k) {
			real AKJ = A[k][j];		
			b[k] =  (k == l) ? b[k]/A[l][j] : b[k]  - bL*AKJ/A[l][j];
			for(int i = 0; i < c.length; ++i) {	
				if (k != l) {
					A[k][i] -= A[l][i]*AKJ/A[l][j];
				}
			}
		}
		real ALJ = A[l][j];		
		for (int i = 0; i < c.length; ++i) {
			A[l][i] = A[l][i]/ALJ;
		}
		finalText += latexCodeForPrintingTheSimplexTableau(c, A, b, currentCost, basicIndices);
	}
	return finalText;
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
	finalText += "\textbf{Step 2.} In order to find a feasible solution, the artificial";
	if (b.length == 1) {
		finalText += "variable $x_" + string(c.length + 1) + "$is introduced";
	}
	if (b.length == 2) {
		finalText += "variables $x_" + string(c.length + 1) + "$ and $x_" + string(c.length + 2) + "$ are introduced"; 
	} 
	if (b.length > 2) {
		finalText += "variables $x_" + string(c.length + 1) + ", \ldots, x_" + string(c.length + b.length) + "$ are introduced ";
	}
	finalText += "to form the following auxiliary problem:";
        real[] c2=concat(array(c.length,0),array(b.length,1));
	real[][] A2 = A; 
        real[][] I = identity(b.length);
	for (int i = 0; i < b.length; ++i) {
          A2[i].append(I[i]);
	}
	finalText += latexCodeforPrintingTheLPP(c2, A2, b) + '\n';

	//----Phase II:
	finalText += "\noindent\textbf{Phase II.}\\" +  '\n\n';
	
	//-Step I.
	finalText += "\textbf{Step 1.} "; 
	int[] a5 = {4,5,6};
	finalText += latexCodeForPrintingSimplexTableaus(c, A, b, a5); 

	return finalText; 
}

file fout = output("out.tex");
write(fout,"\documentclass{article}
\usepackage{amsmath,color}
\begin{document}
");

real[] c = {-3/4,20,-1/2,6,0,0};
real[][] A = {{1/4,-8,-1,9,1,0,0},{1/2,-11,-1/2,3,0,1,0},{0,0,1,0,0,0,1}};
real[] b = {0,0,1};
write(fout,solveLPPUsingTheSimplexMethod(c,A,b));  
write(fout,"\end{document}");

if(settings.verbose > 0) write("Wrote out.tex");

real[] c = {-10,-12,-12,0,0,0};
real[][] A = {{1,2,2,1,0,0},{2,1,2,0,1,0},{2,2,1,0,0,1}};
real[] b = {20,20,20};

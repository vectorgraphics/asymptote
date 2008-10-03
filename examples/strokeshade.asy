size(100);
guide g=(0,0)..controls(70,30) and (-40,30)..(30,0);
latticeshade(g,stroke=true,linewidth(10),
	     new pen[][] {{red,orange,yellow},{green,blue,purple}});

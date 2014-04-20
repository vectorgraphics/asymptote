struct sCAD
{
  int    nLineGroup = 0; // 0-3

  pen
  // A
  pA,
  pVisibleEdge,            // Sichtbare Kanten
  pVisibleContour,        // Sichtbarer Umriss
  pUsableWindingLength,    // Nitzbare Gewindel‰nge
  pSystemLine,            // Systemlinie (Stahlbau)
  pDiagramCurve,            // Kurve in Diagrammen
  pSurfaceStructure,        // Oberfl‰chenstrukturen
  // B
  pB,
  pLightEdge,                // Lichtkante
  pMeasureLine,            // Maﬂlinie
  pMeasureHelpLine,        // Maﬂhilfslinie
  pMeasureLineBound,        // Maﬂlinienbegrenzung
  pReferenceLine,            // Hinweislinie
  pHatch,                    // Schraffur
  pWindingGround,            // Gewindegrund
  pDiagonalCross,            // Diagonalkreuz
  pBendLine,                // Biegelinie
  pProjectionLine,        // Projektionslinie
  pGrid,                    // Rasterlinien
  // C
  pC,
  pFreehand,                // Begrenzung abgebrochener oder unterbrochener
  // Schnitte, wenn die Begrenzung
  // keine Mittellinie ist
  // E
  pE,
  pSurfaceTreatmentAllowed,    // Bereich zul‰ssiger Oberfl‰chenbehandlung
  // F
  pF,
  pInvisibleEdge,            // unsichtbare Kante
  pInvisibleContour,        // unsichtbarer Umriss
  // G
  pG,
  pMiddleLine,            // Mittellinie
  pSymmetryLine,            // Symmetrielinie
  pPartialCircle,            // Teilkreis
  pCircularHole,            // Lochkreis
  pDivisionPlane,            // Teilungsebene
  pTransferLine,            // Trajektorien (‹bertragunslinien)
  // J
  pJ,
  pCuttingPlane,                // Schnittebene
  pSurfaceTreatmentRequested,    // Bereich geforderter Behandlungen
  // K
  pK,
  pContourBeforeDeformation,    // Umrisse vor Verformung
  pAdjacentPartContour,        // Umrisse angrenzender Teile
  pEndShapeRawMaterial,        // Fertigformen in Rohteilen
  pContourEligibleType,        // Umrisse wahlweiser Ausf¸hrungen
  pPartInFrontOfCuttingPlane;    // Teile vor der Schnittebene



  static sCAD Create(int nLineGroup = 1)
  {
    sCAD    cad = new sCAD;
    if ( nLineGroup < 0 )
      nLineGroup = 0;
    if ( nLineGroup > 3 )
      nLineGroup = 3;
    cad.nLineGroup = nLineGroup;

    restricted real[]    dblFullWidth = {0.35mm, 0.5mm, 0.7mm, 1.0mm};
    restricted real[]    dblHalfWidth = {0.18mm, 0.25mm, 0.35mm, 0.5mm};

    pen    pFullWidth = linewidth(dblFullWidth[nLineGroup]);
    pen    pHalfWidth = linewidth(dblHalfWidth[nLineGroup]);

    // Linienarten:
    // A
    cad.pA =
      cad.pVisibleEdge =
      cad.pVisibleContour =
      cad.pUsableWindingLength =
      cad.pSystemLine =
      cad.pDiagramCurve =
      cad.pSurfaceStructure =
      pFullWidth + solid;
    // B
    cad.pB =
      cad.pLightEdge =
      cad.pMeasureLine =
      cad.pMeasureHelpLine =
      cad.pMeasureLineBound =
      cad.pReferenceLine =
      cad.pHatch =
      cad.pWindingGround =
      cad.pDiagonalCross =
      cad.pBendLine =
      cad.pProjectionLine =
      cad.pGrid =
      pHalfWidth + solid;
    // C
    cad.pC =
      cad.pFreehand =
      pHalfWidth + solid;
    // D
    // Missing, as I have no idea how to implement this...
    // E
    cad.pE =
      cad.pSurfaceTreatmentAllowed =
      pFullWidth + linetype(new real[] {10,2.5});
    // F
    cad.pF =
      cad.pInvisibleEdge =
      cad.pInvisibleContour =
      pHalfWidth + linetype(new real[] {20,5});
    // G
    cad.pG =
      cad.pMiddleLine =
      cad.pSymmetryLine =
      cad.pPartialCircle =
      cad.pCircularHole =
      cad.pDivisionPlane =
      cad.pTransferLine =
      pHalfWidth + linetype(new real[] {40,5,5,5});
    // H
    // see J
    // I
    // This letter is not used in DIN 15
    // J
    cad.pJ =
      cad.pCuttingPlane =
      cad.pSurfaceTreatmentRequested =
      pFullWidth + linetype(new real[] {20,2.5,2.5,2.5});
    // K
    cad.pK =
      cad.pContourBeforeDeformation =
      cad.pAdjacentPartContour =
      cad.pEndShapeRawMaterial =
      cad.pContourEligibleType =
      cad.pPartInFrontOfCuttingPlane =
      pHalfWidth + linetype(new real[] {40,5,5,5,5,5});

    return cad;
  } // end of Create



  real    GetMeasurementBoundSize(bool bSmallBound = false)
  {
    if ( bSmallBound )
      return 1.5 * linewidth(pVisibleEdge) / 2;
    else
      return 5 * linewidth(pVisibleEdge);
  }



  path    GetMeasurementBound(bool bSmallBound = false)
  {
    if ( bSmallBound )
      return scale(GetMeasurementBoundSize(bSmallBound = bSmallBound)) *
        unitcircle;
    else
      return (0,0) --
        (-cos(radians(7.5)), -sin(radians(7.5))) *
        GetMeasurementBoundSize(bSmallBound = bSmallBound) --
        (-cos(radians(7.5)), sin(radians(7.5))) *
        GetMeasurementBoundSize(bSmallBound = bSmallBound) --
        cycle;
  }



  void    MeasureLine(picture pic = currentpicture,
                      Label L,
                      pair pFrom,
                      pair pTo,
                      real dblLeft = 0,
                      real dblRight = 0,
                      real dblRelPosition = 0.5,
                      bool bSmallBound = false)
  {
    if ( dblLeft < 0 )
      dblLeft = 0;
    if ( dblRight < 0 )
      dblRight = 0;
    if ( (dblLeft > 0) && (dblRight == 0) )
      dblRight = dblLeft;
    if ( (dblLeft == 0) && (dblRight > 0) )
      dblLeft = dblRight;
    pair    pDiff = pTo - pFrom;
    real    dblLength = length(pDiff);
    pair    pBegin = pFrom - dblLeft * unit(pDiff);
    pair    pEnd = pTo + dblRight * unit(pDiff);
    if ( bSmallBound )
      {
        draw(
             pic = pic,
             g = pBegin--pEnd,
             p = pMeasureLine);
      }
    else
      {
        real dblBoundSize = GetMeasurementBoundSize(bSmallBound = bSmallBound);
        if ( dblLeft == 0 )
          draw(
               pic = pic,
               g = (pFrom + dblBoundSize/2 * unit(pDiff))
               -- (pTo - dblBoundSize/2 * unit(pDiff)),
               p = pMeasureLine);
        else
          draw(
               pic = pic,
               g = pBegin -- (pFrom - dblBoundSize/2 * unit(pDiff))
               ^^ pFrom -- pTo
               ^^ (pTo + dblBoundSize/2 * unit(pDiff)) -- pEnd,
               p = pMeasureLine);
      }
    path    gArrow = GetMeasurementBound(bSmallBound = bSmallBound);
    picture    picL;
    label(picL, L);
    pair    pLabelSize = 1.2 * (max(picL) - min(picL));
    if ( dblLeft == 0 )
      {
        fill(
             pic = pic,
             g = shift(pFrom) * rotate(degrees(-pDiff)) * gArrow,
             p = pVisibleEdge);
        fill(
             pic = pic,
             g = shift(pTo) * rotate(degrees(pDiff)) * gArrow,
             p = pVisibleEdge);
        if ( dblRelPosition < 0 )
          dblRelPosition = 0;
        if ( dblRelPosition > 1 )
          dblRelPosition = 1;
        label(
              pic = pic,
              L = rotate(degrees(pDiff)) * L,
              position =
              pFrom
              + dblRelPosition * pDiff
              + unit(rotate(90)*pDiff) * pLabelSize.y / 2);
      }
    else
      {
        fill(
             pic = pic,
             g = shift(pFrom) * rotate(degrees(pDiff)) * gArrow,
             p = pVisibleEdge);
        fill(
             pic = pic,
             g = shift(pTo) * rotate(degrees(-pDiff)) * gArrow,
             p = pVisibleEdge);
        if ( (dblRelPosition >= 0) && (dblRelPosition <= 1) )
          label(
                pic = pic,
                L = rotate(degrees(pDiff)) * L,
                position =
                pFrom
                + dblRelPosition * pDiff
                + unit(rotate(90)*pDiff) * pLabelSize.y / 2);
        else
          {
            // draw label outside
            if ( dblRelPosition < 0 )
              label(
                    pic = pic,
                    L = rotate(degrees(pDiff)) * L,
                    position =
                    pBegin
                    + pLabelSize.x / 2 * unit(pDiff)
                    + unit(rotate(90)*pDiff) * pLabelSize.y / 2);
            else
              // dblRelPosition > 1
              label(
                    pic = pic,
                    L = rotate(degrees(pDiff)) * L,
                    position =
                    pEnd
                    - pLabelSize.x / 2 * unit(pDiff)
                    + unit(rotate(90)*pDiff) * pLabelSize.y / 2);
          }
      }
  } // end of MeasureLine



  void    MeasureParallel(picture pic = currentpicture,
                          Label L,
                          pair pFrom,
                          pair pTo,
                          real dblDistance,
                          // Variables from MeasureLine
                          real dblLeft = 0,
                          real dblRight = 0,
                          real dblRelPosition = 0.5,
                          bool bSmallBound = false)
  {
    pair    pDiff = pTo - pFrom;
    pair    pPerpendicularDiff = unit(rotate(90) * pDiff);
    real    dblDistancePlus;
    if ( dblDistance >= 0 )
      dblDistancePlus = dblDistance + 1mm;
    else
      dblDistancePlus = dblDistance - 1mm;
    draw(
         pic = pic,
         g = pFrom--(pFrom + dblDistancePlus*pPerpendicularDiff),
         p = pMeasureHelpLine
         );
    draw(
         pic = pic,
         g = pTo--(pTo + dblDistancePlus*pPerpendicularDiff),
         p = pMeasureHelpLine
         );
    MeasureLine(
                pic = pic,
                L = L,
                pFrom = pFrom + dblDistance * pPerpendicularDiff,
                pTo = pTo + dblDistance * pPerpendicularDiff,
                dblLeft = dblLeft,
                dblRight = dblRight,
                dblRelPosition = dblRelPosition,
                bSmallBound = bSmallBound);
  } // end of MeasureParallel



  path MakeFreehand(pair pFrom, pair pTo,
                    real dblRelDivisionLength = 12.5,
                    real dblRelDistortion = 2.5,
                    bool bIncludeTo = true)
  {
    pair    pDiff = pTo - pFrom;
    pair    pPerpendicular = dblRelDistortion * linewidth(pFreehand) *
      unit(rotate(90) * pDiff);

    int nNumOfSubDivisions=ceil(length(pDiff) /
                                (dblRelDivisionLength * linewidth(pFreehand)));

    restricted real[]    dblDistortion = {1, -.5, .75, -.25, .25, -1, .5, -.75,
                                          .25, -.25};
    int                    nDistortion = 0;

    guide    g;
    g = pFrom;
    for ( int i = 1 ; i < nNumOfSubDivisions ; ++i )
      {
        g = g ..
          (pFrom
           + pDiff * i / (real)nNumOfSubDivisions
           + pPerpendicular * dblDistortion[nDistortion]);
        nDistortion += 1;
        if ( nDistortion > 9 )
          nDistortion = 0;
      }

    if ( bIncludeTo )
      g = g .. pTo;

    return g;
  } // end of MakeFreehand



} // end of CAD


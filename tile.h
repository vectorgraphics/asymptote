#pragma once
/*
 * Tile rendering library (API-agnostic).
 *
 * Computes a grid of tiles covering a target image and provides the
 * per-tile projection frustum parameters.  The caller is responsible
 * for setting the projection, rendering the scene, and reading back
 * each tile's pixels into the final image buffer.
 *
 * Basic usage:
 *
 *   TileContext tr;
 *   tr.setImageSize(fullWidth, fullHeight);
 *   tr.setTileSize(tileW, tileH, border);
 *   tr.setOrtho(xmin, xmax, ymin, ymax, zNear, zFar);
 *   // or tr.setFrustum(...)
 *
 *   do {
 *       tr.beginTile();
 *       // set projection via tr.getLeft()/getRight() etc.
 *       // render scene
 *   } while (tr.endTile());
 */

#include <cmath>
#include <cstddef>

namespace camp
{

struct TileContext
{
  // Final image parameters
  int imageWidth = 0;
  int imageHeight = 0;

  // Tile parameters
  int tileWidth = 256;
  int tileHeight = 256;
  int tileBorder = 0;
  int tileWidthNB = 256;  // tile width without border
  int tileHeightNB = 256; // tile height without border

  // Projection parameters
  bool perspective = false;
  double left = 0, right = 0, bottom = 0, top = 0;
  double zNear = 0, zFar = 0;

  // Tile iteration state
  int rows = 0;
  int columns = 0;
  int currentTile = 0;
  int currentRow = 0;
  int currentColumn = 0;
  int currentTileWidth = 0;   // actual rendered width (with border)
  int currentTileHeight = 0;  // actual rendered height (with border)

  // Row order
  enum RowOrder { BOTTOM_TO_TOP, TOP_TO_BOTTOM };
  RowOrder rowOrder = BOTTOM_TO_TOP;

  // Current tile's frustum (set by beginTile)
  double tileLeft = 0, tileRight = 0;
  double tileBottom = 0, tileTop = 0;

  TileContext() { reset(); }

  void reset()
  {
    currentTile = 0;
    rows = 0;
    columns = 0;
  }

  void setTileSize(int width, int height, int border)
  {
    tileBorder = border;
    tileWidth = width;
    tileHeight = height;
    tileWidthNB = width - 2 * border;
    tileHeightNB = height - 2 * border;
    computeGrid();
  }

  void setImageSize(int width, int height)
  {
    imageWidth = width;
    imageHeight = height;
    computeGrid();
  }

  void setOrtho(double l, double r, double b, double t, double nearVal, double farVal)
  {
    perspective = false;
    left = l; right = r; bottom = b; top = t;
    zNear = nearVal; zFar = farVal;
  }

  void setFrustum(double l, double r, double b, double t, double nearVal, double farVal)
  {
    perspective = true;
    left = l; right = r; bottom = b; top = t;
    zNear = nearVal; zFar = farVal;
  }

  void setRowOrder(RowOrder order) { rowOrder = order; }

  // Accessors for current tile's frustum (called after beginTile)
  double getLeft()   const { return tileLeft; }
  double getRight()  const { return tileRight; }
  double getBottom() const { return tileBottom; }
  double getTop()    const { return tileTop; }
  double getZNear()  const { return zNear; }
  double getZFar()   const { return zFar; }
  bool isPerspective() const { return perspective; }

  // Tile geometry (called after beginTile)
  int getCurrentRow()        const { return currentRow; }
  int getCurrentColumn()     const { return currentColumn; }
  int getCurrentTileWidth()  const { return currentTileWidth; }
  int getCurrentTileHeight() const { return currentTileHeight; }
  int getRows()              const { return rows; }
  int getColumns()           const { return columns; }

  // Destination offsets in the final image (no-border region)
  int getDestX() const { return tileWidthNB * currentColumn; }
  int getDestY() const { return tileHeightNB * currentRow; }

  // Source region within the rendered tile (no-border region)
  int getSrcX()    const { return tileBorder; }
  int getSrcY()    const { return tileBorder; }
  int getSrcWidth()  const { return currentTileWidth - 2 * tileBorder; }
  int getSrcHeight() const { return currentTileHeight - 2 * tileBorder; }

  // Begin rendering the next tile. Returns false if no tiles remain.
  bool beginTile()
  {
    if (currentTile == 0)
      computeGrid();

    if (currentTile >= rows * columns)
      return false;

    if (rowOrder == BOTTOM_TO_TOP) {
      currentRow = currentTile / columns;
      currentColumn = currentTile % columns;
    } else {
      currentRow = rows - (currentTile / columns) - 1;
      currentColumn = currentTile % columns;
    }

    // Compute actual tile size (last row/column may be smaller)
    if (currentRow < rows - 1)
      currentTileHeight = tileHeight;
    else
      currentTileHeight = imageHeight - (rows - 1) * tileHeightNB + 2 * tileBorder;

    if (currentColumn < columns - 1)
      currentTileWidth = tileWidth;
    else
      currentTileWidth = imageWidth - (columns - 1) * tileWidthNB + 2 * tileBorder;

    // Compute this tile's frustum
    tileLeft   = left   + (right - left)   * (currentColumn * tileWidthNB - tileBorder) / imageWidth;
    tileRight  = tileLeft + (right - left)   * currentTileWidth / imageWidth;
    tileBottom = bottom + (top - bottom) * (currentRow * tileHeightNB - tileBorder) / imageHeight;
    tileTop    = tileBottom + (top - bottom) * currentTileHeight / imageHeight;

    currentTile++;
    return true;
  }

  // Signal that the current tile is done. Returns true if more tiles remain.
  bool endTile()
  {
    return currentTile < rows * columns;
  }

private:
  void computeGrid()
  {
    if (imageWidth <= 0 || imageHeight <= 0 || tileWidthNB <= 0 || tileHeightNB <= 0) {
      rows = 0;
      columns = 0;
      return;
    }
    columns = (imageWidth + tileWidthNB - 1) / tileWidthNB;
    rows = (imageHeight + tileHeightNB - 1) / tileHeightNB;
  }
};

} // namespace camp


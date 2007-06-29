#!/usr/bin/env python
###########################################################################
#
# Convert a Bezier curve to a polyline
#
# Once Tk supports "RawCurves" this will not be needed.
#
#
# Author: Orest Shardt
# Created: June 29, 2007
#
###########################################################################
import math

def norm(vector):
  """Return the norm of a vector"""
  return math.sqrt(vector[0]**2+vector[1]**2)

def distancePointToLine(point,line0,line1):
  #return the distance from a point to the line segment line0--line1
  
  #find a vector in the direction of the line
  d = (line0[0]-line1[0],line0[1]-line1[1])
  l = norm(d)
  if l == 0:
    return norm((point[0]-line0[0],point[1]-line0[1]))
  u = (-d[1]/l,d[0]/l) #now it's perpendicular
  v = (point[0]-line0[0],point[1]-line0[1])
  #now find proj of v onto u
  #proj_u v = (v.u)*u
  dot = v[0]*u[0]+v[1]*u[1]
  d = (dot*u[0],dot*u[1])
  return norm(d)

def splitLine(end0,end1,t):
  """Split a line at the distance t, with t in (0,1)"""
  return (end0[0]+t*(end1[0]-end0[0]),end0[1]+t*(end1[1]-end0[1]))

def splitBezier(node0,control0,control1,node1,t):
  """Find the nodes and control points for the segments of a Bezier curve split at t"""
  a = splitLine(node0,control0,t)
  b = splitLine(control0,control1,t)
  c = splitLine(control1,node1,t)
  d = splitLine(a,b,t)
  e = splitLine(b,c,t)
  f = splitLine(d,e,t)#this is the point on the curve at t
  return ([node0,a,d,f],[f,e,c,node1])

def BezierWidth(node0,control0,control1,node1):
  """Find a quantity related to the distance of the controls from the node-node line"""
  return distancePointToLine(control0,node0,node1)+distancePointToLine(control1,node0,node1)


#If the above algorithm fails, this one will work, but it is far from elegant
#def computeIntermediates(steps,node0,control0,control1,node1):
  #pointList = []
  #for a in range(0,100,100/steps)+[100]:
   #t = a/100.0
   #t1 = 1-t
   #x = node0[0]*t1**3+3*control0[0]*t*t1**2+3*control1[0]*t**2*t1+node1[0]*t**3
   #y = node0[1]*t1**3+3*control0[1]*t*t1**2+3*control1[1]*t**2*t1+node1[1]*t**3
   #pointList.append((x,y))
  #return pointList
#def makeBezier(steps,node0,control0,control1,node1):
 #if len(node0)!=2 or len(control0)!=2 or len(control1)!=2 or len(node1)!=2:
  #return -1
 #else:
  #return [node0]+computeIntermediates(steps,node0,control0,control1,node1)+[node1]

def makeBezierIntermediates(node0,control0,control1,node1,epsilon):
  """Find the points, excluding node0, to be used as the line segment endpoints"""
  if(BezierWidth(node0,control0,control1,node1) <= epsilon):
    return [node1]
  else:
    splitUp = splitBezier(node0,control0,control1,node1,0.5)
    return makeBezierIntermediates(*splitUp[0]+[epsilon])+makeBezierIntermediates(*splitUp[1]+[epsilon])

def makeBezier(node0,control0,control1,node1,epsilon=1):
  """Return the vertices to be used in the polyline representation of a Bezier curve"""
  return [node0]+makeBezierIntermediates(node0,control0,control1,node1,epsilon)

if __name__ == '__main__':
  pointList = makeBezier((-80,0),(-150,40),(150,120),(80,0),0.5)
  from timeit import Timer
  t = Timer('makeBezier((-80,0),(-40,-40),(40,120),(80,0),1)','from __main__ import makeBezier')
  print pointList
  print len(pointList)
  iterations = 1000
  time = t.timeit(iterations)
  print "%d iterations took %f seconds (%f ms for each)."%(iterations,time,1000.0*time/iterations)
  points = []
  for point in pointList:
    points.append(point[0])
    points.append(-point[1])
  from Tkinter import *
  root = Tk()
  canv = Canvas(root,scrollregion=(-100,-100,100,100))
  canv.pack()
  canv.create_line(points)
  for point in pointList:
   canv.create_oval(point[0],-point[1],point[0],-point[1],fill='red',outline='red')
  root.mainloop()
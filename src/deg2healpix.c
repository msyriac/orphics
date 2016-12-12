#include "chealpix.h"
#include <math.h>
#include<stdio.h>
#include<stdlib.h>

/*

The only slow step in making healpix maps is converting
from ra,dec to healpix pixel index.
The C function getPixIndex here uses the other helper functions
here to do it quickly. makeMap.py uses ctypes to call
this function and make a healpix map in around 3 minutes.

Use scripts/MakeDeg.sh to compile this module into a shared library.

- Mat M.

Some functions from
ftp://ftp.eso.org/scisoft/scisoft4/linux/redhat9/saoimage/saoimage-1.29.3/wcscon.c

*/


/* Transform degrees to radians and back */

double degrad( deg)
     double  deg ;
 {
  return deg*M_PI/180. ;
}


double raddeg(rad)
     double  rad ;
 {
  return rad*180./M_PI ;
}



/* Convert right ascension, declination, and distance to
   geocentric equatorial rectangular coordinates */

void
s2v3 (rra,rdec,r,pos)

     double rra;/* Right ascension in radians */
     double rdec;/* Declination in radians */
     double r;/* Distance to object in same units as pos */
     double pos[3];/* x,y,z geocentric equatorial position of object (returned) */
{
  pos[0] = r * cos (rra) * cos (rdec);
  pos[1] = r * sin (rra) * cos (rdec);
  pos[2] = r * sin (rdec);

  return;
}


/* Convert geocentric equatorial rectangular coordinates to
   right ascension and declination, and distance */

void
v2s3 (pos,rra,rdec,r)

     double pos[3];/* x,y,z geocentric equatorial position of object */
     double *rra;/* Right ascension in radians (returned) */
     double *rdec;/* Declination in radians (returned) */
     double *r;/* Distance to object in same units as pos (returned) */

{
  double x,y,z,rxy,rxy2,z2;

  x = pos[0];
  y = pos[1];
  z = pos[2];

  *rra = atan2 (y, x);
  if (*rra < 0.) *rra = *rra + 6.283185307179586;

  rxy2 = x*x + y*y;
  rxy = sqrt (rxy2);
  *rdec = atan2 (z, rxy);

  z2 = z * z;
  *r = sqrt (rxy2 + z2);

  return;
}

  /*  Equatorial to galactic rotation matrix
      The eulerian angles are p, q, 90-r
      +cp.cq.sr-sp.cr     +sp.cq.sr+cp.cr     -sq.sr
      -cp.cq.cr-sp.sr     -sp.cq.cr+cp.sr     +sq.cr
      +cp.sq              +sp.sq              +cq*/
  
  static
    double jgal[3][3] = {
    -0.054875539726,-0.873437108010,-0.483834985808,
    0.494109453312,-0.444829589425, 0.746982251810,
    -0.867666135858,-0.198076386122, 0.455983795705};


/*--- Transform IAU 1958 galactic coordinates to J2000 equatorial coordinates */

void
gal2fk5 (dtheta,dphi)

     double *dtheta;/* Galactic longitude (l2) in degrees
		       J2000.0 ra in degrees (returned) */
     double *dphi;/* Galactic latitude (b2) in degrees
		     J2000.0 dec in degrees (returned) */

     /*  Note:
       The equatorial coordinates are J2000.  Use the routine FK42GAL
       if conversion to J2000 coordinates is required.
       Reference: Blaauw et al, MNRAS,121,123 (1960) */

{
  double pos[3],pos1[3],r,dl,db,rl,rb,rra,rdec,dra,ddec;
  void v2s3(),s2v3();
  int i;
  char *eqcoor, *eqstrn();

  /*  Spherical to Cartesian */
  dl = *dtheta;
  db = *dphi;


  rl = degrad (dl);
  rb = degrad (db);
  r = 1.0;
  s2v3 (rl,rb,r,pos);

  /*  Rotate to equatorial coordinates */
  for (i = 0; i < 3; i++) {
    pos1[i] = pos[0]*jgal[0][i] + pos[1]*jgal[1][i] + pos[2]*jgal[2][i];
  }

  /*  Cartesian to Spherical */
  v2s3 (pos1,&rra,&rdec,&r);
  dra = raddeg (rra);
  ddec = raddeg (rdec);


  *dtheta = dra;
  *dphi = ddec;

  /* /\*  Print result if in diagnostic mode *\/ */
  /* if (idg) { */
  /*   fprintf (stderr,"GAL2FK5: long = %.5f lat = %.5f\n",dl,db); */
  /*   eqcoor = eqstrn (dra,ddec); */
  /*   fprintf (stderr,"GAL2FK5: J2000 RA,Dec= %s\n",eqcoor); */
  /*   free (eqcoor); */
  /* } */

  return;
}






void
fk52gal (dtheta,dphi)
     
     double *dtheta;/* J2000 right ascension in degrees
		       Galactic longitude (l2) in degrees (returned) */
     double *dphi;/* J2000 declination in degrees
		     Galactic latitude (b2) in degrees (returned) */
     
     /* Rotation matrices by P.T.Wallace, Starlink eqgal and galeq, March 1986 */
     /*  Note:
	 The equatorial coordinates are J2000 FK5.  Use the routine
	 GAL2FK4 if conversion from B1950 FK4 coordinates is required.
	 Reference: Blaauw et al, MNRAS,121,123 (1960) */
{
  double pos[3],pos1[3],r,dl,db,rl,rb,rra,rdec,dra,ddec;
  void v2s3(),s2v3();
  char *eqcoor, *eqstrn();
  int i;
  



  
  /*  Spherical to cartesian */
  dra = *dtheta;
  ddec = *dphi;
  rra = degrad (dra);
  rdec = degrad (ddec);
  r = 1.0;
  (void)s2v3 (rra,rdec,r,pos);


  /*  Rotate to galactic */
  for (i = 0; i < 3; i++) {
    pos1[i] = pos[0]*jgal[i][0] + pos[1]*jgal[i][1] + pos[2]*jgal[i][2];
  }

  /*  Cartesian to spherical */
  v2s3 (pos1,&rl,&rb,&r);

  dl = raddeg (rl);
  db = raddeg (rb);
  *dtheta = dl;
  *dphi = db;

  /*  Print result if in diagnostic mode */
  //eqcoor = eqstrn (dra,ddec);
  //fprintf (stderr,"FK52GAL: J2000 RA,Dec= %s\n",eqcoor);
  //fprintf (stderr,"FK52GAL: long = %.5f lat = %.5f\n",dl,db);
  //free (eqcoor);
  
  
  return;
}


// Convert an ra,dec to healpix index in galactic coordinates
   
long getPixIndexGalactic(const long nside, double ra_degree, double dec_degree)
{

  double long_rad;
  double colat_rad;

  double *ra, *dec;

  long pix;
  long *ipring = &pix;

  long Npix = 12*nside*nside;

  ra = &ra_degree;
  dec = &dec_degree;

  // convert to l,b
  fk52gal(ra,dec) ;

  // validate l,b
  /* if (ra_degree<0. || ra_degree>360.) {printf("\n%s\n","Angle error."); return;} */
  /* if (dec_degree<-90. || dec_degree>90.) {printf("\n%s\n","Angle error."); return;} */

  
  // convert to radians
  long_rad = degrad(ra_degree);
  colat_rad = M_PI/2.0 - degrad(dec_degree);


  // validate radians
  /* if (colat_rad<0. || colat_rad>M_PI) {printf("\n%s\n","Angle error."); return;} */
  /* if (long_rad<0. || long_rad>(2.0*M_PI)) {printf("\n%s\n","Angle error."); return;} */

	  
  // convert to healpix ring index
  ang2pix_ring(nside, colat_rad, long_rad, ipring);


  // validate pixel
  /* if (pix<0 || pix>Npix) {printf("\n%s\n","Pixel index error."); return;} */

  return pix;
}







int RotateIndexGtoC(const long nside, const long pixInd)
{


  double l , b  ;




  double colat_rad;
  double *theta = &colat_rad;

  double long_rad;
  double *phi = &long_rad;


  long pix;
  long *ipring = &pix;

  pix2ang_ring(nside, pixInd, theta, phi);



  // convert to degrees
  l = raddeg(long_rad);
  //b =  raddeg(colat_rad);
  b =  raddeg(M_PI/2.0 - colat_rad);


  double *raOut, *decOut ;
  raOut = &l;
  decOut = &b;

  // convert to l,b
  fk52gal(raOut,decOut) ;


  double thetaPass, phiPass;

  thetaPass = M_PI/2.0 - degrad(*decOut);
  phiPass = degrad(*raOut);


  // convert to healpix ring index
  ang2pix_ring(nside, thetaPass, phiPass, ipring);

  /* printf("%d,", pixInd); */
  /* printf("%d,", pix); */
  /* printf("%.2f,", b); */
  /* printf("%.2f\n", l); */





  return pix;
}


long getPixIndexEquatorial(const long nside, double ra_degree, double dec_degree)
{

  /*
    Pseudo-code

    l,b = fk52gal(ra,dec) 
      long = l
      colat = pi/2 - b
      indG = ang2pix(long,colat)
        long, colat = pix2ang(indG)
        l = long
        b = pi/2 - colat
    x,y = fk52gal(l,b) 
    long = x
    colat = pi/2 - y;
    ingE = ang2pix(long,colat)

   */


  long pix1, pix;

  pix1 = getPixIndexGalactic(nside, ra_degree, dec_degree);
  pix = RotateIndexCtoG(nside,pix1);

  /* long pix; */
  /* double *ra, *dec; */
  /* long *ipring = &pix; */

  /* ra = &ra_degree; */
  /* dec = &dec_degree; */

  /* // convert to l,b */
  /* fk52gal(ra,dec) ; */
  /* // convert to l,b */
  /* fk52gal(ra,dec) ; */


  /* double thetaPass, phiPass; */

  /* thetaPass = M_PI/2.0 - degrad(*dec); */
  /* phiPass = degrad(*ra); */


  /* // convert to healpix ring index */
  /* ang2pix_ring(nside, thetaPass, phiPass, ipring); */

  
  /* printf("%d,", pix2); */
  /* printf("%d,", pix); */

  return pix;


}






/* This function hasn't been tested. */
void RotateMapGtoC(double * hpInMap, double * hpOutMap, const long nside)
{

  


  int pixInd, newIndex;

  const long Npix = 12*nside*nside;



  for ( pixInd=0; pixInd<Npix; pixInd++ )
    {
      
      //printf("%d,",pixInd) ;
      
      newIndex = RotateIndexGtoC(nside, pixInd);
      hpOutMap[pixInd] = hpInMap[newIndex];


    }

}




void getRaDec(const long nside, const long pixInd, double *raDeg, double *decDeg)
{


  double l , b  ;




  double colat_rad;
  double *theta = &colat_rad;

  double long_rad;
  double *phi = &long_rad;


  pix2ang_ring(nside, pixInd, theta, phi);



  // convert to degrees
  l = raddeg(long_rad);
  b =  raddeg(M_PI/2.0 - colat_rad);


  double *raOut, *decOut ;
  raOut = &l;
  decOut = &b;

  // convert l,b to ra, dec
  gal2fk5(raOut,decOut) ;


  *raDeg = *raOut;
  *decDeg = *decOut;




  return;
}



int RotateIndexCtoG(const long nside, const long pixInd)
{


  double l , b  ;




  double colat_rad;
  double *theta = &colat_rad;

  double long_rad;
  double *phi = &long_rad;


  long pix;
  long *ipring = &pix;

  pix2ang_ring(nside, pixInd, theta, phi);



  // convert to degrees
  l = raddeg(long_rad);
  //b =  raddeg(colat_rad);
  b =  raddeg(M_PI/2.0 - colat_rad);


  double *raOut, *decOut ;
  raOut = &l;
  decOut = &b;

  // convert to l,b
  gal2fk5(raOut,decOut) ;


  double thetaPass, phiPass;

  thetaPass = M_PI/2.0 - degrad(*decOut);
  phiPass = degrad(*raOut);


  // convert to healpix ring index
  ang2pix_ring(nside, thetaPass, phiPass, ipring);

  /* printf("%d,", pixInd); */
  /* printf("%d,", pix); */
  /* printf("%.2f,", b); */
  /* printf("%.2f\n", l); */





  return pix;
}

      !program fortran_diagonalization
      !    INTEGER :: NM=3, N=3, IERR
      !    REAL*8 :: D(3)=0.0, E(3)=0.0
      !    REAL*8 :: Z(3,3)=0.0
      !    Z(1,1)=85.02
      !    Z(1,2)=75.57
      !    Z(1,3)=68.28
      !    Z(2,1)=75.57
      !    Z(2,2)=145.66
      !    Z(2,3)=113.83
      !    Z(3,1)=68.28
      !    Z(3,2)=113.83
      !    Z(3,3)=94.35
      !    WRITE (*,*) Z
      !    call TRED2(NM,N,D,E,Z)
      !    WRITE (*,*) 'Z', Z
      !    WRITE (*,*) 'D' ,D
      !    WRITE (*,*) 'E', E
      !    call IMTQL2(NM,N,D,E,Z,IERR)
      !    WRITE (*,*) 'Z', Z
      !    WRITE (*,*) 'D' ,D
      !    WRITE (*,*) 'E', E
      !end program fortran_diagonalization
!********************************************
       SUBROUTINE TRED2(NM,N,D,E,Z)
C     This subroutine is a translation of the ALGOL procedure TRED2,
C     NUM. MATH. 11, 181-195(1968) by Martin, Reinsch, and Wilkinson.
C     HANDBOOK FOR AUTO. COMP., VOL.II-LINEAR ALGEBRA, 212-226(1971).
C
C     This subroutine reduces a REAL SYMMETRIC matrix to a
C     symmetric tridiagonal matrix using and accumulating
C     orthogonal similarity transformations.
C
C     On Input
C
C        NM must be set to the row dimension of the two-dimensional
C          array parameters, A and Z, as declared in the calling
C          program dimension statement.  NM is an INTEGER variable.
C
C        N is the order of the matrix A.  N is an INTEGER variable.
C          N must be less than or equal to NM.
C
C        A contains the real symmetric input matrix.  Only the lower
C          triangle of the matrix need be supplied.  A is a two-
C          dimensional REAL array, dimensioned A(NM,N).
C     On Output
C
C        D contains the diagonal elements of the symmetric tridiagonal
C          matrix.  D is a one-dimensional REAL array, dimensioned D(N).
C
C        E contains the subdiagonal elements of the symmetric
C          tridiagonal matrix in its last N-1 positions.  E(1) is set
C          to zero.  E is a one-dimensional REAL array, dimensioned
C          E(N).
C
C        Z contains the orthogonal transformation matrix produced in
C          the reduction.  Z is a two-dimensional REAL array,
C          dimensioned Z(NM,N).
C
C        A and Z may coincide.  If distinct, A is unaltered.


       IMPLICIT DOUBLE PRECISION(A-H,O-z)
       DIMENSION D(N),E(N),Z(NM,N)
       INTENT(INOUT) Z
       IF(N.EQ.1) GOTO 320
       DO 300 II=2,N
         I=N+2-II
         L=I-1
         H=0.0D0
         SCALE=0.0D0
         IF(L.LT.2) GOTO 130
         DO 120 K=1,L
120      SCALE=SCALE+DABS(Z(I,K))
         IF(SCALE.ne.0.0D0) GOTO 140
130      E(I)=Z(I,L)
         GOTO 290
140      DO 150 K=1,L
           Z(I,K)=Z(I,K)/SCALE
           H=H+Z(I,K)**2
150      CONTINUE
         F=Z(I,L)
         G=-DSIGN(DSQRT(H),F)
         E(I)=SCALE*G
         H=H-F*G
         Z(I,L)=F-G
         F=0.0D0
         DO 240 J=1,L
           Z(J,I)=Z(I,J)/(SCALE*H)
           G=0.0D0
           DO 180 K=1,J
180        G=G+Z(J,K)*Z(I,K)
           JP1=J+1
           IF(L.LT.JP1) GOTO 220
           DO 200 K=JP1,L
200        G=G+Z(K,J)*Z(I,K)
220        E(J)=G/H
           F=F+E(J)*Z(I,J)
240      CONTINUE
         HH=F/(H+H)
         DO 260 J=1,L
           F=Z(I,J)
           G=E(J)-HH*F
           E(J)=G
           DO 260 K=1,J
             Z(J,K)=Z(J,K)-F*E(K)-G*Z(I,K)
260      CONTINUE
         DO 280 K=1,L
280      Z(I,K)=SCALE*Z(I,K)
290      D(I)=H
300   CONTINUE
320   d(1)=0.0D0
      E(1)=0.0D0
      DO 500 I=1,N
        L=I-1
        IF(D(I).eq.0.0d0) GOTO 380
        DO 360 J=1,L
          G=0.0D0
          DO 340 K=1,L
340       G=G+Z(I,K)*Z(K,J)
          DO 360 K=1,L
            Z(K,J)=Z(K,J)-G*Z(K,I)
360     CONTINUE
380     D(I)=Z(I,I)
        Z(I,I)=1.0D0
        IF(L.LT.1) GOTO 500
        DO 400 J=1,L
          Z(I,J)=0.0D0
          Z(J,I)=0.0D0
400     CONTINUE
500   CONTINUE
      RETURN
      END
!****************************************************************
      subroutine imtql2(nm,n,d,e,z,ierr)   
C     This subroutine is a translation of the ALGOL procedure IMTQL2,
C     NUM. MATH. 12, 377-383(1968) by Martin and Wilkinson,
C     as modified in NUM. MATH. 15, 450(1970) by Dubrulle.
C     HANDBOOK FOR AUTO. COMP., VOL.II-LINEAR ALGEBRA, 241-248(1971).
C
C     This subroutine finds the eigenvalues and eigenvectors
C     of a SYMMETRIC TRIDIAGONAL matrix by the implicit QL method.
C     The eigenvectors of a FULL SYMMETRIC matrix can also
C     be found if  TRED2  has been used to reduce this
C     full matrix to tridiagonal form.
C
C     On INPUT
C
C        NM must be set to the row dimension of the two-dimensional
C          array parameter, Z, as declared in the calling program
C          dimension statement.  NM is an INTEGER variable.
C
C        N is the order of the matrix.  N is an INTEGER variable.
C          N must be less than or equal to NM.
C
C        D contains the diagonal elements of the symmetric tridiagonal
C          matrix.  D is a one-dimensional REAL array, dimensioned D(N).
C
C        E contains the subdiagonal elements of the symmetric
C          tridiagonal matrix in its last N-1 positions.  E(1) is
C          arbitrary.  E is a one-dimensional REAL array, dimensioned
C          E(N).
C
C        Z contains the transformation matrix produced in the reduction
C          by  TRED2,  if performed.  This transformation matrix is
C          necessary if you want to obtain the eigenvectors of the full
C          symmetric matrix.  If the eigenvectors of the symmetric
C          tridiagonal matrix are desired, Z must contain the identity
C          matrix.  Z is a two-dimensional REAL array, dimensioned
C          Z(NM,N).
C
C      On OUTPUT
C
C        D contains the eigenvalues in ascending order.  If an
C          error exit is made, the eigenvalues are correct but
C          unordered for indices 1, 2, ..., IERR-1.
C
C        E has been destroyed.
C
C        Z contains orthonormal eigenvectors of the full symmetric
C          or symmetric tridiagonal matrix, depending on what it
C          contained on input.  If an error exit is made,  Z contains
C          the eigenvectors associated with the stored eigenvalues.
C
C        IERR is an INTEGER flag set to
C          Zero       for normal return,
C          J          if the J-th eigenvalue has not been
C                     determined after 30 iterations.
C                     The eigenvalues and eigenvectors should be correct
C                     for indices 1, 2, ..., IERR-1, but the eigenvalues
C                     are not ordered.

      implicit double precision(a-h,o-z)
      dimension d(n),e(n),z(nm,n)
      dmach=2.0d0**(-37)
      ierr=0
      if(n.eq.1)goto 1011
      do 101 i=2,n
 101  e(i-1)=e(i)
      e(n)=0.0d0
      do 240 l=1,n
        j=0
 105    do 110 m=l,n
          if(m.eq.n)goto 120
          if(dabs(e(m)).le.dmach*(dabs(d(m))+dabs(d(m+1))))goto 120
 110    continue
 120    p=d(l)
        if(m.eq.l)goto 240
        if(j.eq.30)goto 1010
        j=j+1
        g=(d(l+1)-p)/(2.0d0*e(l))
        r=dsqrt(g*g+1.0d0)
        g=d(m)-p+e(l)/(g+dsign(r,g))
        s=1.0d0
        c=1.0d0
        p=0.0d0
        mml=m-l
        do 200 ii=1,mml
          i=m-ii
          f=s*e(i)
          b=c*e(i)
          if(dabs(f).lt.dabs(g)) goto 150
          c=g/f
          r=dsqrt(c*c+1.0d0)
          e(i+1)=f*r
          s=1.0d0/r
          c=c*s
          goto 160
 150      s=f/g
          r=dsqrt(s*s+1.0d0)
          e(i+1)=g*r
          c=1.0d0/r
          s=s*c
 160      g=d(i+1)-p
          r=(d(i)-g)*s+2.0d0*c*b
          p=s*r
          d(i+1)=g+p
          g=c*r-b
          do 180 k=1,n
            f=z(k,i+1)
            z(k,i+1)=s*z(k,i)+c*f
            z(k,i)=c*z(k,i)-s*f
 180      continue
 200    continue
        d(l)=d(l)-p
        e(l)=g
        e(m)=0.0d0
        goto 105
 240  continue
      do 300 ii=2,n
        i=ii-1
        k=i
        p=d(I)
        do 260 j=ii,n
          if(d(j).ge.p)goto 260
          k=j
          p=d(j)
 260    continue
        if(k.eq.i) goto 300
        d(k)=d(i)
        d(i)=p
        do 280 ll=1,n
          p=z(ll,i)
          z(ll,i)=z(ll,k)
          z(ll,k)=p
 280    continue
 300  continue
      goto 1011
1010  ierr=l
1011  return
      end
!***********************************************

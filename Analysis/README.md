## how to use this module ::Analysis::

```python
	from Analysis import XYZ
	#for xyz file
	with XYZ("path/sometraj.xyz") as traj:
	    # play with attributes 
        traj.atom_lt # get atom list in this traj
	    traj.AtomNum # get atom number in this traj
	    traj.coords # get atom coords in this first frame
	    for f in traj:
 	        print(f.coords)
	# for zip file
	with Analysis.XYZ("path/sometraj.zip", "trajfile_name.xyz") as trajZ:
		..

	
```

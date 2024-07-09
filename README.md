# Trilogy
Trilogy color images

Automatically convert FITS images into pretty pictures

Most recent example:
* https://github.com/cosmic-spring/Earendel/blob/main/Trilogy%20color%20images%20WHL0137%20Sunrise%20Arc%20Earendel.ipynb

Other examples:
* https://github.com/dancoe/Trilogy/blob/main/Trilogy%20color%20images%20Webb%20focus%20field.ipynb
* https://github.com/dancoe/Trilogy/blob/main/Trilogy%20NIRCam%20color%20images.ipynb
* https://github.com/dancoe/CEERS/blob/main/NIRCam%20Trilogy%20color%20images.ipynb
* https://github.com/dancoe/mirage/blob/main/Trilogy%20color%20images%20NIRCam.ipynb


Recent adaptation to Python 3:
https://github.com/oliveirara/trilogy

Original code: https://www.stsci.edu/~dcoe/trilogy

### Load RGB FITS image into ds9

* Scale -- Min Max
* Frame -- New Frame RGB
* File -- Open as -- RGB Image...

### Load RGB FITS image into APT Aladin

* Aladin
* Cmd-I
* File
* Browse
* RGB fits image
* (3 planes show up in Aladin)
* rgb
  * red = RGB[1]
  * green = RGB[2]
  * blue = RGB[3]

### Load RGB FITS image into Python

```
color_image_hdulist = fits.open(color_image_file)
color_image_wcs = wcs.WCS(color_image_hdulist[0].header, color_image_hdulist)
color_image_data = np.stack([hdu.data for hdu in color_image_hdulist[1:]], axis=-1)
```

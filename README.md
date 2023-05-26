## Hyper-Skin Data

![img](figure/sampledata.png)
Hyper-Skin is a comprehensive hyperspectral dataset designed specifically for facial skin analysis. It covers a wide spectral range from visible (VIS) to near-infrared (NIR) spectra, allowing for a holistic understanding of various aspects of human facial skin. The dataset includes 330 hyperspectral cubes captured from 51 subjects, featuring diverse facial poses and angles. Each hyperspectral cube contains 1,048,576 spectra of length 448, providing a rich and detailed representation of skin properties.  The 448-bands data is resampled into two 31-band datasets: one covering the visible spectrum from 400nm to 700nm, and the other covering the near-infrared spectrum from 700nm to 1000nm. Additionally, synthetic RGB and Multispectral (MSI) data are generated, comprising RGB images and an infrared image at 960nm. These steps result in the creation of our Hyper-Skin dataset, encompassing two types of data: (RGB, VIS) and (MSI, NIR). 

To ensure an unbiased evaluation of our data, we employed a participant-based data splitting approach. Specifically, we randomly selected 10% of the participants from our participant pool and allocated their data exclusively to the testing set. This method ensures that the testing set contains facial skin data from unseen subjects, who were not represented in the training set. By using participant-based data splitting, we mitigate the risk of potential bias and ensure a more robust assessment of the generalization capabilities of our models. 

### Data Download Instructions
The data is provided publicly available in our SharedPoint cloud. 
- Click [Hyper-Skin(RGB, VIS)](https://utoronto.sharepoint.com/:f:/r/sites/fase-hyper-skin/Shared%20Documents/Hyper-Skin(MSI,%20NIR)?csf=1&web=1&e=yoDSDm) to download the (RGB, VIS) data pair.
- Click [Hyper-Skin(MSI, NIR)](https://utoronto.sharepoint.com/:f:/r/sites/fase-hyper-skin/Shared%20Documents/Hyper-Skin(MSI,%20NIR)?csf=1&web=1&e=BGlo2x) to download the (MSI, NIR) data pair.



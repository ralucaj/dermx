# DermX - a dermatological diagnosis explainability dataset

Repository for the DermX dataset. You can find the paper in `DermX.pdf`, and the experiments in 
[the experiments repo](https://github.com/leoilab/dermx-experiments).

## Data structure
The data is split into two folders: `annotations` and `masks`.
All dermatologist annotation are available in the `annotations` folder. Each JSON file is named 
`dermatologist_image-name.json`, and contains labelling information structured as:

```
{
  "dataset": "Explainability dataset",
  "image": {
      // Image metadata
    },
  "annotations": [
    {
      "name": "Characteristic name",
      "bounding_box": {
        // Height, width, and (x, y) of the upper left corner of the bounding box
        // Appears only if the characteristic was localisable
      },
      "complex_polygon": {
        // Array of paths to trace each mask
      },
      "attributes": [
        // Array of additional terms, if the characteristic is a basic terms
      ]
    },
    {
      "name": "Diagnosis or non-localisable characteristic",
      "tag":{}
    }
  ]
}
```

For ease of use, binary masks were also added in the `masks` folder. The `masks/instance_masks_annotations.csv` 
file creates the relationship between the `image_id`, the `mask_id` as found in `masks/masks`, and the characteristic 
name. Masks are stored as black and white PNG images, where black represents the background and white the selected area.

Most images have been renamed during labelling. Their original filename and source dataset can be found in `metadata.csv`.
This file contains the original filename as found in the original dataset, the source dataset, the diagnosis associated 
with the image, and the filename under which the image can be found in DermX.

## Getting access to the original images
The authors have not yet received the right to redistribute the SD and DermNetNZ images. To access the original images, 
please contact the owners of [the SD-260 dataset](http://xiaopingwu.cn/assets/projects/sd-198/), and 
[DermNetNZ dataset](https://dermnetnz.org/contact-us/). Access to this data is only allowed for research purposes.

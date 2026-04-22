import gudhi as gd
import numpy as np
from typing import List, Dict
from .base import TopologicalDistance
from ..representations import PersistenceDiagram, PersistenceImage


class ImageDistance(TopologicalDistance):
    """
    Takes in persistence diagrams, converts them to persistence images, and computes the distance between them.
    This TopologicalDistance subclass supports comparison of distributions (i.e. input of lists of persistence diagrams).

    Attributes
    ----------
    diagram 1 : PersistenceDiagram or List[PersistenceDiagram]
        The PersistenceDiagram or List[PersistenceDiagram] to be compared with diagram2.
    diagram 2 : PersistenceDiagram or List[PersistenceDiagram]
        The PersistenceDiagram or List[PersistenceDiagram] to be compared with diagram1.
    norm : int, default=2.
        Defines what norm will be used for calculations. Default is 2.
    bandwidth : double, default = 1.0
        Controls the Gaussian kernel for the probability distribution calculated for each point on the birth/persistence diagram.
        See gudhi documentation for more information.
    weight : function, default = lambda x: 1
        Defines the weight function used to compute the weighted sum of the probability distributions for each point on the bith/persistence diagram, i.e. the persistence surface.
        Default is a constant function. Other common choices are a linear function or a bump function, which put greater emphasis on points with longer persistence.
        See gudhi documentation for more information.
    resolution : List[int, int], default = [20,20]
        The dimensions of the persistence image in pixels.
    image_transformer : gudhi package object
        The object that powers the transformation of persistence diagrams into persistence images.
    """

    def __init__(
        self,
        diagram1,
        diagram2,
        norm=2,
        bandwidth=1.0,
        weight=lambda x: 1,
        resolution=[20, 20],
    ) -> None:
        super().__init__(diagram1, diagram2, norm)
        self.bandwidth = bandwidth
        self.weight = weight
        self.resolution = resolution
        self.image_transformer = gd.representations.PersistenceImage(
            bandwidth=bandwidth, weight=weight, resolution=resolution
        )

    def supports_distribution(self) -> bool:
        """Indicates support for distributions of persistence images."""
        return True

    #  ╭──────────────────────────────────────────────────────────╮
    #  │ Core Methods: Fit & Transform                            │
    #  ╰──────────────────────────────────────────────────────────╯

    def fit(self):
        """Computes and returns persistence images from the persistence diagrams, diagram1 and diagram2 (or the average persistence images, if diagram1 and/or diagram2 are distributions.

        Returns
        -------
        avg1 : PersistenceImage
            A wrapper object for the (average) persistence landscape for diagram1.
            Attribute called 'pixels' stores a Dict[int, np.array] that a maps homology dimension key to a np.array that stores image data.

        avg2 : PersistenceImage
            A wrapper object for the (average) persistence landscape for diagram1.
            Attribute called 'pixels' stores a Dict[int, np.array] that a maps homology dimension key to a np.array that stores image data.
        """

        """Compute and return average persistence images for both diagrams."""
        img1 = self._convert_to_image(self.diagram1)
        img2 = self._convert_to_image(self.diagram2)

        avg1 = self._average_image(img1)
        avg2 = self._average_image(img2)
        return avg1, avg2

    def transform(self, image1, image2) -> float:
        """Computes the norm-based distance between two persistence images.
        Cannot be executed before fit() method has been run to generate persistence images.

        Parameters
        ----------
        image1 : PersistenceImage
            A wrapper object for the first persistence image to be compared.
            Attribute called 'pixels' stores a Dict[int, np.array] that a maps homology dimension key to a np.array with shape (resolution[0] * resolution[1], 1), which contains the values of the persistence surface at each pixel.
        image2 : PersistenceImage
            A wrapper object for the second persistence image to be compared.
            Attribute called 'pixels' stores a Dict[int, np.array] that a maps homology dimension key to a np.array with shape (resolution[0] * resolution[1], 1), which contains the values of the persistence surface at each pixel.

        Returns
        -------
        float :
            The norm-based distance between the two given persistence landscapes.
        """
        common_dims = set(image1.homology_dims).intersection(image2.homology_dims)
        difference = self._subtract_images(image1, image2)
        distance = sum(self.compute_norm(difference.pixels[dim]) for dim in common_dims)
        return distance

    def fit_transform(self) -> float:
        """Generates persistence images for diagram1, diagram2 and computes the distance between them.
        Returns
        -------
        float :
            The distance between diagram1 and diagram2, computed via image vectorization.
        """
        avg1, avg2 = self.fit()
        return self.transform(avg1, avg2)

    #  ╭──────────────────────────────────────────────────────────╮
    #  │ Helper Functions                                         │
    #  ╰──────────────────────────────────────────────────────────╯

    def _convert_to_image(self, diagrams: List[PersistenceDiagram]) -> List[PersistenceImage]:
        """Convert each persistence diagram in the given list into to a persistence image in the returned list."""
        images = []

        for diagram in diagrams:
            img = PersistenceImage(
                homology_dims=diagram.homology_dims,
                bandwidth=self.bandwidth,
                weight=self.weight,
                resolution=self.resolution,
            )
            pixels = {}
            for dim, points in diagram.persistence_pts.items():
                transformed_points = self.image_transformer.fit_transform([points])[0]
                pixels[dim] = transformed_points
            img.pixels = pixels
            images.append(img)
        return images

    def _average_image(self, images: List[PersistenceImage]) -> PersistenceImage:
        """Compute and return the average persistence image across multiple images."""
        avg_image = ImageDistance._initialize_average_image(images)
        ImageDistance._accumulate_image_pixels(images, avg_image)
        ImageDistance._normalize_image(avg_image, len(images))
        return ImageDistance._create_average_image_object(images, avg_image)

    @staticmethod
    def _initialize_average_image(
        images: List[PersistenceImage],
    ) -> Dict[int, np.ndarray]:
        """Initialize an average image dictionary with zero arrays."""
        avg_image = {}
        for dim in images[0].pixels.keys():
            avg_image[dim] = np.zeros_like(images[0].pixels[dim])
        return avg_image

    @staticmethod
    def _accumulate_image_pixels(
        images: List[PersistenceImage],
        avg_image: Dict[int, np.ndarray],
    ) -> None:
        """Accumulate image pixels into the average image."""
        for image in images:
            for dim, pixels in image.pixels.items():
                avg_image[dim] += pixels

    @staticmethod
    def _normalize_image(
        avg_image: Dict[int, np.ndarray],
        num_images: int,
    ) -> None:
        """Normalize the average image by the number of images."""
        for dim in avg_image:
            avg_image[dim] /= num_images

    @staticmethod
    def _create_average_image_object(
        images: List[PersistenceImage],
        avg_image: Dict[int, np.ndarray],
    ) -> PersistenceImage:
        """Create and return a PersistenceImage object from the average image."""
        avg_img = PersistenceImage(
            homology_dims=list(avg_image.keys()),
            bandwidth=images[0].bandwidth,
            weight=images[0].weight,
            resolution=images[0].resolution,
        )
        avg_img.pixels = avg_image
        return avg_img

    # TODO: check that weight and other parameters are the same between images
    @staticmethod
    def _subtract_images(image1: PersistenceImage, image2: PersistenceImage) -> PersistenceImage:
        """Subtracts two images for each common dimension, returning a PersistenceImage that represents the difference."""
        common_dims = set(image1.homology_dims).intersection(image2.homology_dims)
        assert (
            image1.resolution == image2.resolution
        ), "Cannot subtract images with different resolutions."
        # Initialize PersistenceImage to return
        diff_pi = PersistenceImage(
            homology_dims=common_dims,
            bandwidth=image1.bandwidth,
            weight=image1.weight,
            resolution=image1.resolution,
        )
        # subtraction
        diff_pixels = {}
        for dim in common_dims:
            diff_pixels[dim] = image1.get_pixels_for_dim(dim) - image2.get_pixels_for_dim(dim)
        diff_pi.pixels = diff_pixels
        return diff_pi

    def __str__(self):
        """Return user-friendly string representation of the ImageDistance instance."""
        return f"ImageDistance object between (1) [{self.diagram1}] and (2) [{self.diagram2}]"

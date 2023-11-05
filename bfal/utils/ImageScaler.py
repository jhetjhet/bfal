import imutils

class ImageScaler:
    def __init__(self, image):
        """
        Initialize the ImageScaler with an input image.

        :param image: The input image (OpenCV format).
        """
        self.original_image = image
        self.scaled_image = None
        self.scale_factor = 1.0

    def set_scale_factor(self, scale_factor):
        """
        Set the scaling factor for the image.

        :param scale_factor: The scaling factor (float).
        """
        self.scale_factor = scale_factor

    def scale_image(self):
        """
        Scale the input image based on the current scale factor.
        """
        if self.scale_factor == 1.0:
            self.scaled_image = self.original_image.copy()
        else:
            self.scaled_image = imutils.resize(self.original_image, width=int(self.original_image.shape[1] * self.scale_factor))

    def get_scaled_image(self):
        """
        Get the scaled image.

        :return: The scaled image (OpenCV format).
        """
        return self.scaled_image

    def scale_image_by_factor(self, factor):
        """
        Scale the image by a specified factor.

        :param factor: The scaling factor (float).
        """
        self.scale_factor *= factor
        self.scale_image()

    def scale_image_by_width(self, target_width):
        """
        Scale the image to a target width.

        :param target_width: The desired width (int).
        """
        original_width = self.original_image.shape[1]
        self.scale_factor = target_width / original_width
        self.scale_image()

    def transform_coordinates(self, scaled_coordinates):
        """
        Transform coordinates from the scaled image to the original image.

        :param scaled_coordinates: A tuple (x, y) of coordinates in the scaled image.
        :return: A tuple (x, y) of transformed coordinates in the original image.
        """
        x, y = scaled_coordinates
        x_original = int(x / self.scale_factor)
        y_original = int(y / self.scale_factor)
        return x_original, y_original

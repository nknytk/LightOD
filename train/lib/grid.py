class GridRetriever:
    def __init__(self, img_x, img_y, n_grid_x, n_grid_y):
        self.img_x = img_x
        self.img_y = img_y
        self.n_grid_x = n_grid_x
        self.n_grid_y = n_grid_y
        self.grid_size_x = int(img_x/n_grid_x)
        self.grid_size_y = int(img_y/n_grid_y)

    def grid_position(self, x_min, y_min, x_max, y_max):
        """
        Input:
          オブジェクト位置座標 (x_min. y_min, x_max, y_max)
        Output:
          grid index, grid内での中心点座標位置x, y, オブジェクト大きさw, h
        """
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        x_grid_index = int(x_center/self.grid_size_x)
        y_grid_index = int(y_center/self.grid_size_y)
        grid_index = y_grid_index * self.n_grid_x + x_grid_index
        x_grid_center = (x_center - x_grid_index * self.grid_size_x)/ self.grid_size_x
        y_grid_center = (y_center - y_grid_index * self.grid_size_y)/ self.grid_size_y
        x_size_relative = (x_center - x_min) / self.img_x
        y_size_relative = (y_center - y_min) / self.img_y
        return grid_index, x_grid_index, y_grid_index, x_grid_center, y_grid_center, x_size_relative, y_size_relative


    def restore_box(self, grid_index, x_grid_center, y_grid_center, x_size, y_size):
        """
        Input:
          grid index, grid内での中心点座標位置x, y, オブジェクト大きさw, h
        Output:
          オブジェクト位置座標 (x_min. y_min, x_max, y_max)
        """
        y_grid_index = int(grid_index / self.n_grid_x)
        x_grid_index = grid_index % self.n_grid_y
        x_center = int((x_grid_index + x_grid_center) * self.grid_size_x)
        y_center = int((y_grid_index + y_grid_center) * self.grid_size_y)
        x_min = x_center - int(x_size * self.img_x)
        y_min = y_center - int(y_size * self.img_y)
        x_max = x_center + int(x_size * self.img_x)
        y_max = y_center + int(y_size * self.img_y)
        return x_min, y_min, x_max, y_max

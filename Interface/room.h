#include <memory>
#include <string>
#include <type_traits>
#include <vector>

class dist_map {
private:
  int direction;
  int x, y, z;
  int height;
  int width;
  std::vector<std::vector<int, char *>> distances;

public:
  dist_map(int direction, int x, int y, int z, int height, int wigth);
  void set_dist_map(std::vector<std::vector<int, char *>> bias);
  void update(std::shared_ptr<dist_map> cover_dist_map);
  std::vector<std::vector<int, char *>> get_dist_map();
};

class item {
private:
  char *item_name;
  char *item_description;
  std::vector<std::shared_ptr<dist_map>> round_dist;

public:
  item(char *name, char *description,
       std::vector<std::shared_ptr<dist_map>> round_dist);
  std::shared_ptr<dist_map> get_dist(int index);
};

class plane_map {
private:
  int plane_loc;
  char *carry;
  char *description;
  std::shared_ptr<dist_map> distance;
  std::vector<std::shared_ptr<item>> items;

public:
  plane_map(int plane_loc, char *carry, char *description,
            std::shared_ptr<dist_map> distance);
  void add_item(std::shared_ptr<item> new_item);
  void update_distances(std::shared_ptr<dist_map> item_dist_map);
};

class dir_map {
private:
  std::vector<std::shared_ptr<plane_map>> plane_maps;

public:
  dir_map(char **charrys, char **descriptions);
  void add_item(std::shared_ptr<item> new_item);
  int judge_plane_map(char *description);
  void update_plane_map(int plane, std::shared_ptr<item> new_item);
  int add_plane(std::shared_ptr<item> new_item);
  std::shared_ptr<plane_map> find_bias(int new_plane_loc);
  void update(std::shared_ptr<plane_map> new_plane);
};

class total_map {
private:
  std::shared_ptr<dir_map> Z_dir_map;
  std::shared_ptr<dir_map> X_dir_map;
  std::shared_ptr<dir_map> Y_dir_map;

public:
  total_map();
  void add_item(std::shared_ptr<item> new_item);
  int judge_dir_map(char *description);
  void update_dir_map(int direction, std::shared_ptr<item> new_item);
  void update_total_map(std::shared_ptr<item> new_item);
};
#include "room.h"
#define sample_points 128

dist_map::dist_map(int dirtion, int x, int y, int z, int height, int wigth) {}
void dist_map::set_dist_map(std::vector<std::vector<int, char *>> bias) {}
void dist_map::update(std::shared_ptr<dist_map> cover_dist_map) {}
std::vector<std::vector<int, char *>> dist_map::get_dist_map() {
  return distances;
}

item::item(char *name, char *description,
           std::vector<std::shared_ptr<dist_map>> round_dist) {}
std::shared_ptr<dist_map> item::get_dist(int index) {
  return round_dist[index];
}

plane_map::plane_map(int plane_loc, char *carry, char *description,
                     std::shared_ptr<dist_map> distance) {}
void plane_map::add_item(std::shared_ptr<item> new_item) {}
void plane_map::update_distances(std::shared_ptr<dist_map> item_dist_map) {}

dir_map::dir_map(char **charrys, char **descriptions) {}
void dir_map::add_item(std::shared_ptr<item> new_item) {}
int dir_map::judge_plane_map(char *description) {}
void dir_map::update_plane_map(int plane, std::shared_ptr<item> new_item) {}
int dir_map::add_plane(std::shared_ptr<item> new_item) {}
std::shared_ptr<plane_map> dir_map::find_bias(int new_plane_loc) {}
void dir_map::update(std::shared_ptr<plane_map> new_plane) {}

total_map::total_map() {}
void total_map::add_item(std::shared_ptr<item> new_item) {}
int total_map::judge_dir_map(char *description) {}
void total_map::update_dir_map(int dirtion, std::shared_ptr<item> new_item) {}
void total_map::update_total_map(std::shared_ptr<item> new_item) {}

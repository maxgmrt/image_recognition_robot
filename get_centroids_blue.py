import utils_max
from utils_max import *

def get_centroids_blue_operators(frame_hist_eq):
    #print("\nBLUE OBJECTS")


    # ISOLATING BLUE ELEMENTS
    mask_blue, res_blue = mask_color(frame_hist_eq)

    # REGION GROWING
    points_done_app = []
    reg_size_app = []

    points_done_app, reg_size_app = region_growing_objects(mask_blue)


    # POINTS FILTERING
    all_points_filtered = []
    reg_size_filtered = []
    #print("%s objects have been detected" % len(points_done_app))

    # we filter objects with less than six points
    for j in range(len(points_done_app)):
        if len(points_done_app[j]) > 6:
            all_points_filtered.append(points_done_app[j])
            reg_size_filtered.append(reg_size_app[j])


    #print("After filtering, %s objects remain" % len(all_points_filtered))
    #print("\nSizes after filtering: %s" % reg_size_filtered)

    centroids = []
    for k in range(len(all_points_filtered)):   
        array = np.asarray(all_points_filtered[k])
        np.reshape(array, (2*len(all_points_filtered[k]), 1)) 
        centroids.append(centeroidnp(array))

    centroids_array = np.asarray(centroids)
    #print("\nObjects centroids:")
    #print(centroids_array)


    final_centroids = centroids_array.tolist()

    index = []

    index = get_index_unwanted_objects(final_centroids)

    #print("\nIndex of removed objects: %s" % index)


    # Division and Equality operators

    for i in range(len(index)-1):
        if index[i] + 1 == index[i+1]:
            index_division = index[i]
        else:
            index_equality = index[i]


    centroid_division = final_centroids[index_division]
    centroid_equality = final_centroids[index_equality]

    unwanted_centroids = []
    unwanted_sizes = []

    for i in range(len(reg_size_filtered)):
        for j in index:
            if (i == j):
                unwanted_centroids.append(final_centroids[i])
                unwanted_sizes.append(reg_size_filtered[i])

    #print("\nUnwanted centroids are: \n%s" % unwanted_centroids)
    #print("Unwanted sizes are: %s" % unwanted_sizes)


    reg_size_final = [e for e in reg_size_filtered if e not in unwanted_sizes]
    final_centroids = [e for e in final_centroids if e not in unwanted_centroids]


    #print("\nFinal sizes: %s" % reg_size_final)
    #print("\nFinal centroids: %s" % final_centroids)


    final_centroids_array = np.asarray(final_centroids)
    reg_size_final_array = np.asarray(reg_size_final)

    sorted_indices_reg = reg_size_final_array.argsort()[::-1]

    # multiplication is the operator with the biggest area
    # addition is the second one
    index_multiplication = sorted_indices_reg[0]
    index_addition = sorted_indices_reg[1]

    centroid_multiplication = final_centroids_array[index_multiplication]
    centroid_addition = final_centroids_array[index_addition]
    centroid_substraction = []


    #print("\n")
    #print("Centroid of multiplication: %s" % centroid_multiplication)
    #print("Centroid of addition: %s" % centroid_addition)
    #print("Centroid of division: %s" % centroid_division)
    #print("Centroid of equality: %s" % centroid_equality)

    return centroid_multiplication, centroid_addition, centroid_division, centroid_equality, centroid_substraction
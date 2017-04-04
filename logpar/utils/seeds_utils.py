def cols_from_csv(filename, cols=None, as_type=str):
    ''' Expects a csv file which first two lines are header '''
    with open(filename) as csv:
        csv.readline()
        csv.readline()
        divided = [line.split(' ') for line in csv]
    if cols:
        points = [[as_type(d[i]) for i in cols] for d in divided if len(d)>1]
    else:
        points = [map(as_type, d) for d in divided if len(d)>1]
    return points


def starting_points(seeds_file):
    ''' Returns starting points grouped by seeds '''
    starting_pnts = cols_from_csv(seeds_file)

    grouped_points = []

    px, py, pz, pstruct, pvi, pvj, pvk = starting_pnts[0]
    pt_group = [tuple(map(float, [px,py,pz]))]
    for px, py, pz, struct, vi, vj, vk in starting_pnts[1:]:
        if struct == pstruct and  (vi, vj, vk) == (pvi, pvj, pvk):
            pt_group.append(tuple(map(float, [px,py,pz])))
        else:
            grouped_points.append(pt_group)
            pt_group = [tuple(map(float, [px,py,pz]))]
            pstruct, pvi, pvj, pvk = struct, vi, vj, vk
    grouped_points.append(pt_group)  # Append final group
    return grouped_points


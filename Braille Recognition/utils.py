import cv2

global global_img_debug #copy the global variable

def get_distance(p1, p2):
    return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2

def get_dot_nearest(dots, diameter, pt1):
    nearest = None
    min_dist = float('inf')
    tolerance = (diameter * 1.5) ** 2  # tolerance was 1.25, changed for more robustness
    for dot in dots:
        dist = get_distance(dot[0], pt1)
        if dist <= tolerance and dist < min_dist:
            nearest = dot
            min_dist = dist
    return nearest

def get_combination(box, dots, diameter):
    global global_img_debug

    result = [0, 0, 0, 0, 0, 0]
    left, right, top, bottom = box
    midpointY = (bottom - top) // 2
    end = (right, midpointY)
    start = (left, midpointY)
    width = right - left

    corners = {
        (left, top): 1,
        (left, top + midpointY): 2,
        (left, bottom): 3,
        (right, top): 4,
        (right, top + midpointY): 5,
        (right, bottom): 6
    }

    local_dots = list(dots)  # Don't mutate original
    for corner, pos in corners.items():
        if global_img_debug is not None:
            cv2.circle(global_img_debug, corner, 6, (0, 0, 255), -1)

        print(f"👉 Checking corner: {corner}, assigned pos {pos}")
        D = get_dot_nearest(local_dots, diameter, corner)
        if D is not None:
            print(f"✅ Found dot near {corner}: {D}")
            local_dots.remove(D)
            result[pos - 1] = 1
        else:
            print(f"❌ No dot near {corner}")
        if not local_dots:
            print("🚫 No more dots left to match.")
            break

    print("🧪 Final result array (dot combo):", result, "| Types:", [type(v) for v in result])
    return end, start, width, tuple(result)


def bounding_box(points):
    """
    points: list of tuples [(x1, y1), (x2, y2), ...]
    return: list of 4 tuples representing rectangle corners
    """
    if not points:
        return []

    # نفصل الإحداثيات X و Y
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    # نحدد الحد الأدنى والأقصى لكل محور
    min_x = min(xs)-2
    max_x = max(xs)+2
    min_y = min(ys)-2
    max_y = max(ys)+2

    # نرجع النقاط الأربعة للمستطيل (حسب الترتيب: أسفل يسار، أسفل يمين، أعلى يمين، أعلى يسار)
    return [
        (min_x, min_y),  # أسفل يسار
        (max_x, min_y),  # أسفل يمين
        (max_x, max_y),  # أعلى يمين
        (min_x, max_y),  # أعلى يسار
    ]

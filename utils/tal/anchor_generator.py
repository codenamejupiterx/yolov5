import torch

from utils.general import check_version

TORCH_1_10 = check_version(torch.__version__, '1.10.0')


@torch.no_grad()
def generate_anchors(feats, strides, grid_cell_size=5.0, grid_cell_offset=0.5, device='cpu', is_eval=False):
    """Generate anchors from features."""
    anchor_points = []
    stride_tensor = []
    assert feats is not None
    dtype = feats[0].dtype
    if is_eval:
        for i, stride in enumerate(strides):
            _, _, h, w = feats[i].shape
            sx = torch.arange(end=w, device=device) + grid_cell_offset  # shift x
            sy = torch.arange(end=h, device=device) + grid_cell_offset  # shift y
            sy, sx = torch.meshgrid(sy, sx, indexing='ij') if TORCH_1_10 else torch.meshgrid(sy, sx)
            anchor_points.append(torch.stack([sx, sy], -1).to(dtype).view(-1, 2))
            stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
        return torch.cat(anchor_points), torch.cat(stride_tensor)
    else:
        anchors = []
        num_anchors_list = []
        for i, stride in enumerate(strides):
            _, _, h, w = feats[i].shape
            cell_half_size = grid_cell_size * stride * 0.5
            sx = (torch.arange(end=w, device=device) + grid_cell_offset) * stride
            sy = (torch.arange(end=h, device=device) + grid_cell_offset) * stride
            sy, sx = torch.meshgrid(sy, sx, indexing='ij') if TORCH_1_10 else torch.meshgrid(sy, sx)
            anchor = torch.stack([sx - cell_half_size, sy - cell_half_size,
                                  sx + cell_half_size, sy + cell_half_size], -1).to(dtype)
            anchors.append(anchor.view(-1, 4))
            anchor_points.append(torch.stack([sx, sy], -1).to(dtype).view(-1, 2))
            num_anchors_list.append(len(anchors[-1]))
            stride_tensor.append(torch.full([num_anchors_list[-1], 1], stride, dtype=dtype, device=device))
        return torch.cat(anchors), torch.cat(anchor_points), num_anchors_list, torch.cat(stride_tensor)


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = torch.split(distance, 2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)  # xywh bbox
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox


def bbox2dist(anchor_points, bbox, reg_max):
    """Transform bbox(xyxy) to dist(ltrb)."""
    x1y1, x2y2 = torch.split(bbox, 2, -1)
    return torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1).clamp(0, reg_max - 0.01)  # dist (lt, rb)

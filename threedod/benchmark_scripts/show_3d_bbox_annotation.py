import vtk
import json
import numpy as np
import argparse
import sys
import subprocess
from plyfile import PlyData


class Render(object):
    def __init__(self, ply_file, json_file, back_face_cull=False):
        """
        :param ply_file: path of ply file
        :param json_file: path of annotation result json file
        :param back_face_cull: see single side of mesh
        """
        self.annotation = load_json(json_file)
        self.file = ply_file
        self.back_face_cull = back_face_cull
        self.reader = vtk.vtkPLYReader()
        self.colors = vtk.vtkNamedColors()
        self.mapper = vtk.vtkPolyDataMapper()
        self.actor = vtk.vtkActor()
        self.ren = vtk.vtkRenderer()
        self.renWin = vtk.vtkRenderWindow()
        self.iren = vtk.vtkRenderWindowInteractor()
        self.offset_x, self.offset_y, self.offset_z = 0, 0, 0
        self.vertex = []
        self.file_type = None

    def __call__(self):
        self._prepare()
        self.iren.Initialize()
        self.renWin.Render()
        self.ren.GetActiveCamera().SetPosition(15.0, 10.0, 9.0)
        self.ren.GetActiveCamera().SetViewUp(0.1, 0.0, 1.0)
        self.renWin.Render()
        self.iren.Start()

    def _prepare(self):
        print("Reading file...")
        self.read_mesh()
        self.set_mapper()
        self.set_actor()
        self.transform_actor()
        self.set_render()
        self.add_actor()
        self.draw_lines()
        self.init_coordinate_axes()
        print("Done")

    def read_mesh(self):
        plydata = None
        file_type = check_file_type(self.file)
        if not file_type:
            plydata = PlyData.read(self.file)
            self.file_type = "pcd" if plydata["face"].count == 0 else "mesh"
        else:
            self.file_type = file_type

        if self.file_type == "mesh":
            self.reader = vtk.vtkPLYReader()
            self.reader.SetFileName(self.file)
            self.reader.Update()
        else:
            if not plydata:
                plydata = PlyData.read(self.file)
            self.vertex = plydata["vertex"]

    def set_mapper(self):
        if self.file_type == "mesh":
            self.mapper.SetInputConnection(self.reader.GetOutputPort())
            self.mapper.SetScalarVisibility(3)
        else:
            points = vtk.vtkPoints()
            vertices = vtk.vtkCellArray()
            polydata = vtk.vtkPolyData()
            for index, vertex in enumerate(self.vertex):
                points.InsertPoint(index, vertex[0], vertex[1], vertex[2])
                vertices.InsertNextCell(1)
                vertices.InsertCellPoint(index)
            polydata.SetPoints(points)
            polydata.SetVerts(vertices)
            self.mapper.SetInputData(polydata)

    def set_actor(self):
        if self.file_type == "mesh":
            self.actor.GetProperty().SetBackfaceCulling(self.back_face_cull)
        else:
            self.actor.GetProperty().SetPointSize(1.5)

        self.actor.SetMapper(self.mapper)
        self.actor.GetProperty().SetColor(self.colors.GetColor3d('Tan'))
        # Place the mesh at the origin point for easy viewing
        self.offset_x = -sum(self.actor.GetXRange()) / 2
        self.offset_y = -sum(self.actor.GetYRange()) / 2
        self.offset_z = -sum(self.actor.GetZRange()) / 2
        self.actor.SetPosition(self.offset_x, self.offset_y, self.offset_z)

    def transform_actor(self):
        # no transformation is required in 3D tool,
        # self.xz_align_matrix is a identity matrix
        self.actor.SetUserMatrix(self.xz_align_matrix)

    def set_render(self):
        self.renWin.SetWindowName("demo")
        self.renWin.SetSize(2500, 1800)
        self.renWin.AddRenderer(self.ren)
        self.ren.SetBackground(self.colors.GetColor3d('AliceBlue'))
        self.ren.GetActiveCamera().SetPosition(15.0, 10.0, 9.0)
        self.ren.GetActiveCamera().SetViewUp(0.1, 0.0, 1.0)
        self.bind_mouse_event()
        self.iren.SetRenderWindow(self.renWin)

    def bind_mouse_event(self):
        self.iren.SetInteractorStyle(MyEvent())

    def add_actor(self):
        self.ren.AddActor(self.actor)

    def init_coordinate_axes(self):
        axes = vtk.vtkAxesActor()
        axes.SetTotalLength(10, 10, 10)
        axes.SetShaftType(0)
        axes.SetCylinderRadius(0.002)
        self.ren.AddActor(axes)

    def draw_lines(self):
        for bbox in self.bboxes:
            self.draw_bbox(bbox)

    def draw_bbox(self, bbox):
        for point in bbox:
            point[0] += self.offset_x
            point[1] += self.offset_y
            point[2] += self.offset_z
        self.ren.AddActor(line_actor([bbox[0], bbox[1], bbox[2], bbox[3],
                                      bbox[0], bbox[4], bbox[5], bbox[6],
                                      bbox[7], bbox[4]]))
        self.ren.AddActor(line_actor([bbox[3], bbox[7]]))
        self.ren.AddActor(line_actor([bbox[1], bbox[5]]))
        self.ren.AddActor(line_actor([bbox[2], bbox[6]]))

    @property
    def bboxes(self):
        bbox_list = []
        for label_info in self.annotation["data"]:
            rotation = np.array(label_info["segments"]["obbAligned"]["normalizedAxes"]).reshape(3, 3)
            transform = np.array(label_info["segments"]["obbAligned"]["centroid"]).reshape(-1, 3)
            scale = np.array(label_info["segments"]["obbAligned"]["axesLengths"]).reshape(-1, 3)
            box3d = compute_box_3d(scale.reshape(3).tolist(), transform, rotation)
            bbox_list.append(box3d)
        bbox_list = np.asarray(bbox_list)
        return bbox_list

    @property
    def xz_align_matrix(self):
        # no transformation is required in 3D tool,
        # just return a identity matrix here
        transM = np.identity(4)
        m = [x for y in transM for x in y]
        mat = vtk.vtkMatrix4x4()
        mat.DeepCopy(m)
        mat.Transpose()
        return mat


class MyEvent(vtk.vtkInteractorStyleTrackballCamera):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.AddObserver("MiddleButtonPressEvent", self.middle_button_press)
        self.AddObserver("MiddleButtonReleaseEvent", self.middle_button_release)
        self.AddObserver("LeftButtonPressEvent", self.left_button_press)
        self.AddObserver("LeftButtonReleaseEvent", self.left_button_release)
        self.AddObserver("RightButtonPressEvent", self.right_button_press)
        self.AddObserver("RightButtonReleaseEvent", self.right_button_release)

    def middle_button_press(self, obj, event):
        # print("Middle Button pressed")
        self.OnMiddleButtonDown()
        return

    def middle_button_release(self, obj, event):
        # print("Middle Button released")
        self.OnMiddleButtonUp()
        return

    def left_button_press(self, obj, event):
        # print("Left Button pressed")
        self.OnLeftButtonDown()
        return

    def left_button_release(self, obj, event):
        # print("Left Button released")
        self.OnLeftButtonUp()
        return

    def right_button_press(self, obj, event):
        # print("right Button pressed")
        self.OnRightButtonDown()
        return

    def right_button_release(self, obj, event):
        # print("right Button released")
        self.OnLeftButtonUp()
        return


def load_json(js_path):
    with open(js_path, "r") as f:
        json_data = json.load(f)
    return json_data


def compute_box_3d(scale, transform, rotation):
    scales = [i / 2 for i in scale]
    l, h, w = scales
    center = np.reshape(transform, (-1, 3))
    center = center.reshape(3)
    x_corners = [l, l, -l, -l, l, l, -l, -l]
    y_corners = [h, -h, -h, h, h, -h, -h, h]
    z_corners = [w, w, w, w, -w, -w, -w, -w]
    corners_3d = np.dot(np.transpose(rotation),
                        np.vstack([x_corners, y_corners, z_corners]))

    corners_3d[0, :] += center[0]
    corners_3d[1, :] += center[1]
    corners_3d[2, :] += center[2]
    bbox3d_raw = np.transpose(corners_3d)
    return bbox3d_raw


def check_file_type(file):
    file_type = None
    cmd = f'head -n 30 {file}'
    try:
        p = subprocess.Popen(cmd, shell=True,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT)
        res = [i.strip() for i in p.stdout.readlines()]
    except Exception:
        return file_type
    for i in res:
        try:
            line = i.decode("utf-8")
        except Exception:
            pass
        else:
            if "element face" in line:
                face_count = int(line.split(" ")[-1])
                if face_count == 0:
                    file_type = "pcd"
                else:
                    file_type = "mesh"
                break
    return file_type


def line_actor(points):
    linesPolyData = vtk.vtkPolyData()
    pts = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    namedColors = vtk.vtkNamedColors()

    for point in points:
        pts.InsertNextPoint(point)
    linesPolyData.SetPoints(pts)

    for i in range(len(points) - 1):
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, i)
        line.GetPointIds().SetId(1, i + 1)
        lines.InsertNextCell(line)

    linesPolyData.SetLines(lines)

    # Setup the visualization pipeline
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(linesPolyData)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetLineWidth(4)
    actor.GetProperty().SetColor(namedColors.GetColor3d("Tomato"))
    return actor


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", type=str, help="path of ply file")
    parser.add_argument("-a", "--anno", type=str, help="path of json file")
    parser.add_argument("-s", "--side", type=int, default=1,
                        help="0: double side, 1:single side")
    return parser.parse_args()


if __name__ == '__main__':
    if len(sys.argv) == 1:
        sys.argv.append("-h")
    args = get_args()
    render = Render(args.file, args.anno, args.side)
    render()

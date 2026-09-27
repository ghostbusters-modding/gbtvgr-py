"""A headless rasteriser for probe cubemaps: vertex-coloured world triangles
drawn into an FBO from a standalone moderngl context."""
# Copyright (C) 2026 Colin Sullivan and contributors
# SPDX-License-Identifier: GPL-2.0-only
import numpy as np

from .mathutil import F32, look_at, perspective

VS = """
#version 330
uniform mat4 mvp;
in vec3 in_pos;
in vec4 in_color;
out vec4 v_color;
void main() {
    gl_Position = mvp * vec4(in_pos, 1.0);
    v_color = in_color;
}
"""

FS = """
#version 330
in vec4 v_color;
out vec4 f_color;
void main() {
    f_color = vec4(v_color.rgb, 1.0);
}
"""

# D3D cube face order: +X, -X, +Y, -Y, +Z, -Z, each as (look direction, up)
CUBE_FACES = (((1, 0, 0), (0, 1, 0)), ((-1, 0, 0), (0, 1, 0)),
              ((0, 1, 0), (0, 0, -1)), ((0, -1, 0), (0, 0, 1)),
              ((0, 0, 1), (0, 1, 0)), ((0, 0, -1), (0, 1, 0)))


class Rasteriser:
    def __init__(self):
        self.ctx = None
        self.prog = None
        self.vao = None
        self._buffers = []
        self._fbo = None
        self._fbo_size = 0
        self.count = 0

    def _ensure(self):
        if self.ctx is not None:
            return
        try:
            import moderngl
            self.ctx = moderngl.create_standalone_context()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError('no offscreen GL context: %s' % exc)
        self.prog = self.ctx.program(vertex_shader=VS, fragment_shader=FS)
        self.ctx.enable(moderngl.DEPTH_TEST)

    def set_geometry(self, positions, colors, indices):
        self._ensure()
        self._release_geometry()
        pos = np.ascontiguousarray(np.asarray(positions, F32).reshape(-1, 3))
        col = np.ascontiguousarray(np.asarray(colors, np.uint8).reshape(-1, 4))
        idx = np.ascontiguousarray(np.asarray(indices, np.uint32).reshape(-1))
        self.count = len(idx)
        if self.count == 0:
            return
        vbo = self.ctx.buffer(pos.tobytes())
        cbo = self.ctx.buffer(col.tobytes())
        ibo = self.ctx.buffer(idx.tobytes())
        self._buffers = [vbo, cbo, ibo]
        self.vao = self.ctx.vertex_array(self.prog, [(vbo, '3f', 'in_pos'), (cbo, '4f1', 'in_color')],
                                         ibo)

    def _release_geometry(self):
        if self.vao is not None:
            self.vao.release()
            self.vao = None
        for b in self._buffers:
            b.release()
        self._buffers = []
        self.count = 0

    def _fbo_for(self, size):
        if self._fbo is None or self._fbo_size != size:
            if self._fbo is not None:
                self._fbo.release()
            color = self.ctx.texture((size, size), 4)
            depth = self.ctx.depth_renderbuffer((size, size))
            self._fbo = self.ctx.framebuffer(color_attachments=[color], depth_attachment=depth)
            self._fbo_size = size
        return self._fbo

    def render(self, view, proj, size):
        """(size, size, 4) uint8, rows top-down, black where nothing is drawn."""
        self._ensure()
        fbo = self._fbo_for(size)
        fbo.use()
        fbo.clear(0.0, 0.0, 0.0, 1.0, depth=1.0)
        if self.vao is not None and self.count:
            mvp = (np.asarray(proj, F32) @ np.asarray(view, F32)).astype(F32)
            self.prog['mvp'].write(np.ascontiguousarray(mvp.T).tobytes())
            self.vao.render()
        data = fbo.read(components=4, dtype='f1')
        img = np.frombuffer(data, dtype=np.uint8).reshape(size, size, 4)
        return img[::-1].copy()

    def cubemap(self, pos, size=64):
        """Six faces in D3D order (+X, -X, +Y, -Y, +Z, -Z), 90 degree views."""
        pos = np.asarray(pos, np.float64)
        proj = perspective(90.0, 1.0, 0.1, 20000.0)
        faces = []
        for look, up in CUBE_FACES:
            view = look_at(pos, pos + np.asarray(look, np.float64), up)
            faces.append(self.render(view, proj, size))
        return faces

    def close(self):
        if self.ctx is None:
            return
        self._release_geometry()
        if self._fbo is not None:
            self._fbo.release()
            self._fbo = None
        self.prog.release()
        self.ctx.release()
        self.ctx = None

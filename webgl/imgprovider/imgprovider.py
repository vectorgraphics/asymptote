#!/usr/bin/env python3
from typing import Optional, Awaitable
from tornado.web import Application, RequestHandler
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
from tornado.web import GZipContentEncoding, HTTPError

import OpenEXR
import struct

class EXRHandler(RequestHandler):
    def initialize(self, **kwargs):
        pass

    def prepare(self) -> Optional[Awaitable[None]]:
        self.add_header('Access-Control-Allow-Origin', '*')

    def get(self, prefix, modifier=None):
        modifier_txt = None
        if modifier == 'diffuse':
            modifier_txt = 'diffuse'
        elif modifier == 'refl':
            step = self.get_arguments('step')
            if len(step) == 0:
                raise HTTPError(510)
            else:
                modifier_txt='refl_0.200_{0}'.format(step[0])

        fil = '{0}_{1}.exr'.format(prefix,modifier_txt) \
            if modifier_txt else '{0}.exr'.format(prefix)

        exr_file =  OpenEXR.InputFile(
            './out_dir/{0}'.format(fil))
        window = exr_file.header()['dataWindow']
        width = window.max.x+1
        height = window.max.y+1
        print(exr_file.header())
        r_chan, g_chan, b_chan = exr_file.channels('RGB')
        img_array = []

        for i in range(width*height):
            img_array.extend(
                (struct.unpack('<f', r_chan[4*i: 4*i+4])[0],
                 struct.unpack('<f', g_chan[4*i: 4*i+4])[0],
                 struct.unpack('<f', b_chan[4*i: 4*i+4])[0]
            ))

        self.out_header = {
            'width': width,
            'height': height,
            'data': img_array
        }

        # if 'gzip' in self.encode:
            # self.add_header('Content-Encoding', 'gzip')
        self.write(self.out_header)


def main():
    main_application = Application([
        (r"/([\w]+)/([\w]*)", EXRHandler)
    ])
    main_application.add_transform(GZipContentEncoding)
    main_server = HTTPServer(main_application, decompress_request=True)
    # Drop any root permissions
    main_server.bind(12345)

    main_server.start()
    IOLoop.current().start()


if __name__ == '__main__':
    main()

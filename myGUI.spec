# -*- mode: python -*-
import sys
sys.setrecursionlimit(1000000)
block_cipher = None


a = Analysis(['myGUI.py'],
             pathex=['F:\\DOC\\Anaconda_3510\\02.Practice\\P04.CodeRefactoring180612'],
             binaries=[],
             datas=[],
             hiddenimports=['sklearn.tree._utils','sklearn.neighbors.typedefs','sklearn.neighbors.ball_tree','sklearn.neighbors.dist_metrics','sklearn.neighbors.quad_tree'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          name='myGUI',
          debug=False,
          strip=False,
          upx=True,
          runtime_tmpdir=None,
          console=False )

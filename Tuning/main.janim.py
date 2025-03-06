from janim.imports import *

textFonts = (
    "Fira Sans Regular",
    "Source Han Sans SC VF",
)

typstPreamble = """
#set text(font = (""))
"""

class DividingOctaveScene ( Timeline ):
    def construct ( self ):
        numberLineBuff = 0.1
        numberLineLength = 12
        numberLinePosition = np.array (( 0, 1.6, 0 ))
        auxNumberLinePosition = np.array (( 0, -1.4, 0 ))
        
        
        mob_note = TypstText ( 
            "注：图中采用的坐标轴均为对数轴", 
        )
        mob_numberLine = NumberLine ( 
            width = numberLineLength, 
            x_range = ( -numberLineBuff, 1 + numberLineBuff ) 
        )
        mob_note.points.to_border ( UL, buff = 0.25 )
        mob_numberLine.points.move_to ( numberLinePosition )
        
        
        self.play (
            Write ( mob_note, duration = 0.5 ),
            Create ( mob_numberLine, duration = 1 ),
        )
        
        self.play ( Wait ( 1 ) )
        

if __name__ == "__main__":
    import subprocess
    subprocess.run ( [
        "janim", "run", __file__, "-i",
    ] )
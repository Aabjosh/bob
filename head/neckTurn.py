# turns the head to a specific angle, PWM?

from gpiozero import AngularServo

PIN = 18
MIN_ANGLE = -90
MAX_ANGLE = 90

class Neck:
    def __init__( self, pin=PIN, minAngle=MIN_ANGLE, maxAngle=MAX_ANGLE ):
        self.pin = pin
        self.minAngle = minAngle
        self.maxAngle = maxAngle
        self.position = 0.0
        self.servo = AngularServo( pin, minAngle, maxAngle )
        self.servo.angle = self.position
        print( "zeroed neck..." )

    def resetPosition( self ):
        self.position = 0.0
        self.servo.angle = Neck.position
        print( "zeroed neck..." )

    def moveToAngle( self, angle ):
        if MIN_ANGLE <= angle <= MAX_ANGLE:
            self.servo.angle = angle
            self.position = angle
            print( f"moved to {angle}..." )

    
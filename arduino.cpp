
// '0' means turn on vibrate
// '1' means turn on red LED
// '2' means turn on buzzer
// '3' means clear, turn off vibrate, red LED, and buzzer

// '4' means turn on blue LED
// '5' means turn off blue LED


// vibrate is on pin 4
// blue LED is on pin 13
// red LED is on pin 12
// buzzer is on pin 7

int value = 12;
unsigned long startTimeMillis;

void setup() 
{
  Serial.begin(9600);
  pinMode(4, OUTPUT);
  pinMode(7, OUTPUT);
  pinMode(12, OUTPUT);
  pinMode(13, OUTPUT);
  Serial.setTimeout(1);
  Serial.println("Connection established...");
}

void loop() 
{
  value = (int)Serial.read();
  if(value != -1)
  //while (Serial.available())
   {
    //value = Serial.read();
    startTimeMillis = millis();
    
    if(value == 0)
    {
      digitalWrite(4, HIGH);
    }
    else if(value == 1)
    {
      // turn on red LED
      digitalWrite(12, HIGH);
    }
    else if(value == 2)
    {
      // turn on buzzer
      digitalWrite(7, HIGH);
    }
    else if(value == 4)
    {
      // turn on blue LED
      digitalWrite(13, HIGH);
    }
    else if(value == 5)
    {
      // turn off blue LED
      digitalWrite(13, LOW);
    }
    else if(value == 3)
    {
      // Turn off vibrate
      digitalWrite(4, LOW);
  
      // Turn off Red LED
      digitalWrite(12, LOW);
    
      // Turn off buzzer
      digitalWrite(7, LOW);
    }

   }
}



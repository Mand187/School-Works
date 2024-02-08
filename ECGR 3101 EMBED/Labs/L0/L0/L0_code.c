//*****************************************************************************
//
// Author: Matthew Anderson
// Author: Ali Altabtabae
// Date:   9/21/2023
// Due Date: 9/25/2023
//
// Description:
// This Program simply turns an LED off for a duration if both switch 1 and switch 2 are logical high
//
//*****************************************************************************

#include <stdint.h>
#include <stdbool.h>
#include "inc/hw_memmap.h"
#include "driverlib/debug.h"
#include "driverlib/gpio.h"
#include "driverlib/sysctl.h"

int clkFreq = 16000000; // set clock frequency for timer 


#define SYSTEM_CLOCK_FREQUENCY clkFreq


int main(void) {
    uint8_t switch1Val, pin4Val;  // Variables to store the values of the input pins

    // Enable Port F and E which contains the pins we are interested in
    SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOF);
    SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOE);

    // Check if the peripheral access is enabled
    while (!SysCtlPeripheralReady(SYSCTL_PERIPH_GPIOF))
    {
    }

    // Enable the GPIO pin for the blue LED (PF2). Set the direction as output.
    GPIOPinTypeGPIOOutput(GPIO_PORTF_BASE, GPIO_PIN_2);

    // Enable the GPIO pin for switch 1 (PF4/SW1). Set the direction as input.
    GPIOPinTypeGPIOInput(GPIO_PORTF_BASE, GPIO_PIN_4);
    GPIOPadConfigSet(GPIO_PORTF_BASE, GPIO_PIN_4, GPIO_STRENGTH_2MA, GPIO_PIN_TYPE_STD_WPU);

    // Enable the GPIO pin for the new pin 4 (PE4). Set the direction as input.
    GPIOPinTypeGPIOInput(GPIO_PORTE_BASE, GPIO_PIN_4);
    GPIOPadConfigSet(GPIO_PORTE_BASE, GPIO_PIN_4, GPIO_STRENGTH_2MA, GPIO_PIN_TYPE_STD_WPU);

    uint32_t delayValue = 5 * clkFreq/3;  // assuming a 16 MHz clock 

    //// Loop forever.
    while (1) {

        // Read the values of switch 1 (PF4) and pin 4 (PE4).
        switch1Val = GPIOPinRead(GPIO_PORTF_BASE, GPIO_PIN_4);
        pin4Val = GPIOPinRead(GPIO_PORTE_BASE, GPIO_PIN_4);

        // Check the state of switch 1 and pin 4
        if ((switch1Val != GPIO_PIN_4) && (pin4Val != GPIO_PIN_4)) {
            // Turn off the blue LED (PF2).
            GPIOPinWrite(GPIO_PORTF_BASE, GPIO_PIN_2, 0x0);
            SysCtlDelay(delayValue);

        }
        else {
            // Turn on the blue LED (PF2)
            GPIOPinWrite(GPIO_PORTF_BASE, GPIO_PIN_2, GPIO_PIN_2);
        }
    }
}

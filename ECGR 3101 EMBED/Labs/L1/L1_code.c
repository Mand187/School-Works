//*****************************************************************************
//
// ECGR 3101 LAB 1 VER 0.5
//
// Author: Matthew Anderson
// Author: Ali Altabtabae
// Date:   10/31/2023
// Due Date: 11/03/2023
//
// Description:
// Uses SysTick timer and a external breadboard switch to set time LED blinks
//
//*****************************************************************************

#include <stdint.h>
#include <stdbool.h>

#include "inc/hw_memmap.h"
#include "inc/hw_types.h"
#include "driverlib/gpio.h"
#include "driverlib/sysctl.h"
#include "driverlib/systick.h"
#include "inc/tm4c123gh6pm.h"

void SysTickDelay(unsigned int time) {

    unsigned int counter;
    const uint32_t one_sec = 10*(1e6) ;

    // Calculated loops based on 0.2s delay at DelaySetup().
    for(counter = 0; counter < time; counter++) {
        // Disable SysTick during setup
        NVIC_ST_CTRL_R = 0;
        NVIC_ST_RELOAD_R = one_sec;
        // Clear values written to CURRENT.
        NVIC_ST_CURRENT_R = 0;
        // Enable SysTick
        SysTickEnable();
        // Trigger Systick timer until reload value reaches 0.
        while((NVIC_ST_CTRL_R & (1<<16) ) == 0){}
        NVIC_ST_CTRL_R = 0;
        SysTickDisable();
    }
};

void GPIOPortEInit(){
    // Enable Port E
    SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOE);

    // Check if the peripheral access is enabled
    while (!SysCtlPeripheralReady(SYSCTL_PERIPH_GPIOE)){}

    // Enable the GPIO pin for pin 4 (PE4). Set the direction as input.
    GPIOPinTypeGPIOInput(GPIO_PORTE_BASE, GPIO_PIN_4);
    GPIOPadConfigSet(GPIO_PORTE_BASE, GPIO_PIN_4, GPIO_STRENGTH_2MA, GPIO_PIN_TYPE_STD_WPU);
}


void GPIOPortFInit() {
    // Enable Port F
    SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOF);

    // Check if the peripheral access is enabled
    while (!SysCtlPeripheralReady(SYSCTL_PERIPH_GPIOF)){}

    // Enable the GPIO pin for switch 1 221). Set the direction as input.
    GPIOPinTypeGPIOInput(GPIO_PORTF_BASE, GPIO_PIN_4);
    GPIOPadConfigSet(GPIO_PORTF_BASE, GPIO_PIN_4, GPIO_STRENGTH_2MA, GPIO_PIN_TYPE_STD_WPU);
}

void RGBInit() {
    //Enable RGB pins
    GPIOPinTypeGPIOOutput(GPIO_PORTF_BASE, GPIO_PIN_1);
    GPIOPinTypeGPIOOutput(GPIO_PORTF_BASE, GPIO_PIN_2);
    GPIOPinTypeGPIOOutput(GPIO_PORTF_BASE, GPIO_PIN_3);
}

int main(void) {

    GPIOPortEInit();
    GPIOPortFInit();
    RGBInit();

    SysCtlClockSet(SYSCTL_SYSDIV_10 | SYSCTL_USE_PLL | SYSCTL_OSC_MAIN | SYSCTL_XTAL_16MHZ);

    uint8_t switch1Val, pin4Val;  // Variables to store the values of the input pins

    while (1) {
        // Read the values of switch 1 (PF4) and pin 4 (PE4).
        switch1Val = GPIOPinRead(GPIO_PORTF_BASE, GPIO_PIN_4);
        pin4Val = GPIOPinRead(GPIO_PORTE_BASE, GPIO_PIN_4);

        // Check the state of switch 1 and pin 4
        if ((switch1Val != GPIO_PIN_4) || (pin4Val != GPIO_PIN_4)) {
            // Ensure the blue is off LED (PF2).
            GPIOPinWrite(GPIO_PORTF_BASE, GPIO_PIN_2, 0x0);
            // Turn on Green LED (For Debugging Purposes)
            // 2.5 Second Loop
            GPIOPinWrite(GPIO_PORTF_BASE, GPIO_PIN_2, GPIO_PIN_2);
            SysTickDelay(5);
            GPIOPinWrite(GPIO_PORTF_BASE, GPIO_PIN_2, 0x0);
            SysTickDelay(5);
        }
        else {
            // 10 Second Loop
            GPIOPinWrite(GPIO_PORTF_BASE, GPIO_PIN_2, GPIO_PIN_2);
            SysTickDelay(10);
            GPIOPinWrite(GPIO_PORTF_BASE, GPIO_PIN_2, 0x0);
            SysTickDelay(10);
        }

    }

}

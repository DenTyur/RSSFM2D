use ndarray::prelude::*;
use ndarray_npy::{ReadNpyError, ReadNpyExt, WriteNpyError, WriteNpyExt};
use num_complex::Complex;
use rayon::prelude::*;
use std::f64::consts::PI;
use std::fs::File;
use std::io::BufWriter;

use crate::parameters;
use parameters::Xspace;

pub struct Field2D {
    pub amplitude: f64,
    pub omega: f64,
    pub x_envelop: f64,
}

impl Field2D {
    pub fn new(amplitude: f64, omega: f64, x_envelop: f64) -> Self {
        Self {
            amplitude,
            omega,
            x_envelop,
        }
    }
    pub fn electric_field_time_dependence(&self, t: f64) -> f64 {
        // Возвращает электрическое поле в момент времени t вдоль
        // каждой из пространственных осей x0, x1 и т.д.: массив размерности dim.
        // Каждый элемент этого массива содержит электрическое поле
        // в момент времени t вдоль соответствующей оси.
        // Например, E0 = electric_fielf(2.)[0] - электрическое
        // поле в момент времени t=2 вдоль оси x0.

        let mut electric_field: f64 = 0.;

        if PI / self.omega - t > 0. {
            electric_field = -self.amplitude * f64::sin(self.omega * t).powi(2);
        }
        electric_field
    }

    pub fn field_x_envelop(&self, x: f64) -> f64 {
        // Пространственная огибающая электрического поля вдоль каждой из осей.
        f64::cos(PI / 2. * x / self.x_envelop).powi(2)
    }

    pub fn integrated_field_x_envelop(&self, x: f64) -> f64 {
        0.5 * x + 0.25 * self.x_envelop * 2. / PI * f64::sin(PI * x / self.x_envelop)
    }

    pub fn electric_field(&self, t: f64, x: f64) -> [f64; 2] {
        // Электрическое поле вдоль каждой пространственной оси в момент времени t.
        [
            self.electric_field_time_dependence(t) * self.field_x_envelop(x),
            self.electric_field_time_dependence(t) * self.field_x_envelop(x),
        ]
    }

    pub fn potential_as_array(&self, t: f64, x: &Xspace) -> [Array<f64, Ix1>; 2] {
        // Потенциал электрического поля вдоль каждой из осей.
        // В рассматриваемом случае оси независимы (2 одномерных электрона).
        // Поэтому можно интегрировать в 1D.

        let time_part: f64 = self.electric_field_time_dependence(t);
        let mut space_part: [Array<f64, Ix1>; 2] = [x.grid[0].clone(), x.grid[0].clone()];

        // Электрическое поле отлично от нуля в области пространства:
        // -x_envelop < x < x_envelop (*)
        // В этой области пространства производная потенциала этого поля отлична
        // от нуля. За пределами этой области потенциал -- константа, которая равна
        // потенциалу на границах области (*)
        for i in 0..2 {
            space_part[i].par_iter_mut().for_each(|elem| match *elem {
                x if x <= -self.x_envelop => {
                    *elem = self.integrated_field_x_envelop(-self.x_envelop)
                }
                x if x >= self.x_envelop => *elem = self.integrated_field_x_envelop(self.x_envelop),
                _ => *elem = self.integrated_field_x_envelop(*elem),
            });
        }
        [
            -time_part * space_part[0].clone(),
            -time_part * space_part[1].clone(),
        ]
    }

    pub fn _save_potential(&self, path: &str, x: &Xspace, t: f64) -> Result<(), WriteNpyError> {
        let writer = BufWriter::new(File::create(path)?);
        self.potential_as_array(t, x)[0].write_npy(writer)?;
        Ok(())
    }
}

// pub struct Field1D {
//     pub amplitude: f64,
//     pub omega: f64,
//     pub x_envelop: f64,
// }
//
// impl Field1D {
//     pub fn new(amplitude: f64, omega: f64, x_envelop: f64) -> Self {
//         Self {
//             amplitude,
//             omega,
//             x_envelop,
//         }
//     }
//
//     // Временная зависимость электрического поля
//     pub fn electric_field_time_dependence(&self, t: f64) -> f64 {
//         let mut electric_field: f64 = 0.;
//
//         if PI / self.omega - t > 0. {
//             electric_field = -self.amplitude * f64::sin(self.omega * t).powi(2);
//         }
//         electric_field
//     }
//
//     // Пространственная огибающая электрического поля
//     pub fn field_x_envelop(&self, x: f64) -> f64 {
//         f64::cos(PI / 2. * x / self.x_envelop).powi(2)
//     }
//
//     // Проинтегрированная пространственная огибающая электрического поля
//     pub fn integrated_field_x_envelop(&self, x: f64) -> f64 {
//         0.5 * x + 0.25 * self.x_envelop * 2. / PI * f64::sin(PI * x / self.x_envelop)
//     }
//
//     // Электрическое поле в точке x в момент t
//     pub fn electric_field(&self, t: f64, x: f64) -> f64 {
//         self.electric_field_time_dependence(t) * self.field_x_envelop(x)
//     }
//
//     // Потенциал электрического поля координатный массив в момент времени t
//     pub fn potential_as_array(&self, t: f64, x: &Array<f64, Ix1>) -> Array<f64, Ix1> {
//         let time_part: f64 = self.electric_field_time_dependence(t);
//         let mut space_part: Array<f64, Ix1> = x.clone();
//
//         // Электрическое поле отлично от нуля в области пространства:
//         // -x_envelop < x < x_envelop (*)
//         // В этой области пространства производная потенциала этого поля отлична
//         // от нуля. За пределами этой области потенциал -- константа, которая равна
//         // потенциалу на границах области (*)
//         space_part.par_iter_mut().for_each(|elem| match *elem {
//             x if x <= -self.x_envelop => *elem = self.integrated_field_x_envelop(-self.x_envelop),
//             x if x >= self.x_envelop => *elem = self.integrated_field_x_envelop(self.x_envelop),
//             _ => *elem = self.integrated_field_x_envelop(*elem),
//         });
//         -time_part * space_part
//     }
// }

// intersection_library.h

#ifndef INTERSECTION_LIBRARY_H
#define INTERSECTION_LIBRARY_H

#include <cmath>
#include <tuple>
#include <vector>
#include <limits>

namespace IntersectionLibrary {

// Simple Vector3 struct using standard library
struct Vector3 {
    double x;
    double y;
    double z;

    // Constructors
    Vector3() : x(0.0), y(0.0), z(0.0) {}
    Vector3(double xVal, double yVal, double zVal) : x(xVal), y(yVal), z(zVal) {}

    // Vector addition
    Vector3 operator+(const Vector3& other) const {
        return Vector3(x + other.x, y + other.y, z + other.z);
    }

    // Vector subtraction
    Vector3 operator-(const Vector3& other) const {
        return Vector3(x - other.x, y - other.y, z - other.z);
    }

    // Scalar multiplication
    Vector3 operator*(double scalar) const {
        return Vector3(x * scalar, y * scalar, z * scalar);
    }

    // Scalar division
    Vector3 operator/(double scalar) const {
        return Vector3(x / scalar, y / scalar, z / scalar);
    }

    // Dot product
    double dot(const Vector3& other) const {
        return x * other.x + y * other.y + z * other.z;
    }

    // Cross product
    Vector3 cross(const Vector3& other) const {
        return Vector3(
            y * other.z - z * other.y,
            z * other.x - x * other.z,
            x * other.y - y * other.x);
    }

    // Norm (magnitude) of the vector
    double norm() const {
        return std::sqrt(x * x + y * y + z * z);
    }

    // Normalize the vector
    Vector3 normalized() const {
        double n = norm();
        if (n > std::numeric_limits<double>::epsilon()) {
            return (*this) / n;
        } else {
            return Vector3(0.0, 0.0, 0.0);
        }
    }

    // Component-wise absolute value
    Vector3 abs() const {
        return Vector3(std::abs(x), std::abs(y), std::abs(z));
    }

    // Zero vector
    static Vector3 Zero() {
        return Vector3(0.0, 0.0, 0.0);
    }
};

using Scalar = double;
using IntersectionResult = std::tuple<bool, Vector3>;

/**
 * @brief Computes the intersection point of a line with a plane.
 *
 * The line is defined by two points, and the plane is defined by its normal vector and a point on the plane.
 *
 * @param linePoint1 First point on the line.
 * @param linePoint2 Second point on the line.
 * @param planePoint A point on the plane.
 * @param planeNormal The normal vector of the plane.
 * @return A tuple containing a bool indicating if an intersection exists and the intersection point.
 */
IntersectionResult intersectLinePlane(
    const Vector3& linePoint1,
    const Vector3& linePoint2,
    const Vector3& planePoint,
    const Vector3& planeNormal)
{
    Vector3 lineDir = linePoint2 - linePoint1;
    Scalar denom = planeNormal.dot(lineDir);

    const Scalar EPSILON = std::numeric_limits<Scalar>::epsilon();

    if (std::abs(denom) < EPSILON) {
        return std::make_tuple(false, Vector3::Zero());
    }

    Scalar t = planeNormal.dot(planePoint - linePoint1) / denom;
    Vector3 intersectionPoint = linePoint1 + lineDir * t;

    return std::make_tuple(true, intersectionPoint);
}

/**
 * @brief Computes the intersection points of a line with a sphere.
 */
std::vector<Vector3> intersectLineSphere(
    const Vector3& linePoint1,
    const Vector3& linePoint2,
    const Vector3& sphereCenter,
    Scalar radius)
{
    Vector3 lineDir = linePoint2 - linePoint1;
    Vector3 diff = linePoint1 - sphereCenter;

    Scalar a = lineDir.dot(lineDir);
    Scalar b = 2 * lineDir.dot(diff);
    Scalar c = diff.dot(diff) - radius * radius;

    Scalar discriminant = b * b - 4 * a * c;

    std::vector<Vector3> intersections;

    const Scalar EPSILON = std::numeric_limits<Scalar>::epsilon();

    if (discriminant < -EPSILON) {
        return intersections;
    } else if (std::abs(discriminant) < EPSILON) {
        Scalar t = -b / (2 * a);
        intersections.push_back(linePoint1 + lineDir * t);
    } else {
        Scalar sqrtDiscriminant = std::sqrt(discriminant);
        Scalar t1 = (-b + sqrtDiscriminant) / (2 * a);
        Scalar t2 = (-b - sqrtDiscriminant) / (2 * a);
        intersections.push_back(linePoint1 + lineDir * t1);
        intersections.push_back(linePoint1 + lineDir * t2);
    }

    return intersections;
}

/**
 * @brief Computes the intersection points of a line with an infinite cylinder.
 */
std::vector<Vector3> intersectLineCylinder(
    const Vector3& linePoint1,
    const Vector3& linePoint2,
    const Vector3& cylinderPoint1,
    const Vector3& cylinderPoint2,
    Scalar radius)
{
    Vector3 lineDir = linePoint2 - linePoint1;
    Vector3 cylAxisDir = (cylinderPoint2 - cylinderPoint1).normalized();

    Vector3 d = lineDir - cylAxisDir * lineDir.dot(cylAxisDir);
    Vector3 m = linePoint1 - cylinderPoint1 - cylAxisDir * (linePoint1 - cylinderPoint1).dot(cylAxisDir);

    Scalar a = d.dot(d);
    Scalar b = 2 * d.dot(m);
    Scalar c = m.dot(m) - radius * radius;

    Scalar discriminant = b * b - 4 * a * c;

    std::vector<Vector3> intersections;

    const Scalar EPSILON = std::numeric_limits<Scalar>::epsilon();

    if (std::abs(a) < EPSILON) {
        if (c > EPSILON) {
            return intersections;
        } else if (std::abs(c) < EPSILON) {
            intersections.push_back(linePoint1);
            intersections.push_back(linePoint2);
            return intersections;
        } else {
            intersections.push_back(linePoint1);
            intersections.push_back(linePoint2);
            return intersections;
        }
    }

    if (discriminant < -EPSILON) {
        return intersections;
    } else if (std::abs(discriminant) < EPSILON) {
        Scalar t = -b / (2 * a);
        intersections.push_back(linePoint1 + lineDir * t);
    } else {
        Scalar sqrtDiscriminant = std::sqrt(discriminant);
        Scalar t1 = (-b + sqrtDiscriminant) / (2 * a);
        Scalar t2 = (-b - sqrtDiscriminant) / (2 * a);
        intersections.push_back(linePoint1 + lineDir * t1);
        intersections.push_back(linePoint1 + lineDir * t2);
    }

    return intersections;
}

/**
 * @brief Computes the intersection point of a line with a finite plane (rectangle).
 */
IntersectionResult intersectLineFinitePlane(
    const Vector3& linePoint1,
    const Vector3& linePoint2,
    const Vector3& planeCenter,
    const Vector3& planeNormal,
    const Vector3& planeEdge1,
    const Vector3& planeEdge2,
    Scalar extent1,
    Scalar extent2)
{
    Vector3 lineDir = linePoint2 - linePoint1;
    Scalar denom = planeNormal.dot(lineDir);

    const Scalar EPSILON = std::numeric_limits<Scalar>::epsilon();

    if (std::abs(denom) < EPSILON) {
        return std::make_tuple(false, Vector3::Zero());
    }

    Scalar t = planeNormal.dot(planeCenter - linePoint1) / denom;
    Vector3 intersectionPoint = linePoint1 + lineDir * t;

    Vector3 localPoint = intersectionPoint - planeCenter;

    Scalar projEdge1 = localPoint.dot(planeEdge1.normalized());
    Scalar projEdge2 = localPoint.dot(planeEdge2.normalized());

    if (std::abs(projEdge1) <= extent1 && std::abs(projEdge2) <= extent2) {
        return std::make_tuple(true, intersectionPoint);
    } else {
        return std::make_tuple(false, Vector3::Zero());
    }
}

/**
 * @brief Checks for intersection of a line with a disk (circular cap).
 */
IntersectionResult intersectLineDisk(
    const Vector3& linePoint1,
    const Vector3& linePoint2,
    const Vector3& diskCenter,
    const Vector3& diskNormal,
    Scalar radius)
{
    Vector3 lineDir = linePoint2 - linePoint1;
    Scalar denom = diskNormal.dot(lineDir);

    const Scalar EPSILON = std::numeric_limits<Scalar>::epsilon();

    if (std::abs(denom) < EPSILON) {
        return std::make_tuple(false, Vector3::Zero());
    }

    Scalar t = diskNormal.dot(diskCenter - linePoint1) / denom;
    Vector3 intersectionPoint = linePoint1 + lineDir * t;

    Vector3 diff = intersectionPoint - diskCenter;
    if (diff.norm() <= radius) {
        return std::make_tuple(true, intersectionPoint);
    } else {
        return std::make_tuple(false, Vector3::Zero());
    }
}

/**
 * @brief Computes the intersection points of a line with a finite cylinder, including its caps.
 */
std::vector<Vector3> intersectLineFiniteCylinder(
    const Vector3& linePoint1,
    const Vector3& linePoint2,
    const Vector3& cylinderBase,
    const Vector3& cylinderAxis,
    Scalar height,
    Scalar radius)
{
    Vector3 lineDir = linePoint2 - linePoint1;
    Vector3 cylAxisNorm = cylinderAxis.normalized();

    Vector3 d = lineDir - cylAxisNorm * lineDir.dot(cylAxisNorm);
    Vector3 m = linePoint1 - cylinderBase - cylAxisNorm * (linePoint1 - cylinderBase).dot(cylAxisNorm);

    Scalar a = d.dot(d);
    Scalar b = 2 * d.dot(m);
    Scalar c = m.dot(m) - radius * radius;

    Scalar discriminant = b * b - 4 * a * c;

    std::vector<Vector3> intersections;

    const Scalar EPSILON = std::numeric_limits<Scalar>::epsilon();

    if (std::abs(a) < EPSILON) {
        // Line is parallel to the cylinder axis
        // Optionally, check if the line is inside the cylinder (c <= 0)
        // Here, we return no intersection (empty vector)
        return intersections;
    } else if (discriminant < -EPSILON) {
        // No intersection with the infinite cylinder
        return intersections;
    } else if (discriminant >= -EPSILON) {
        Scalar sqrtDiscriminant = std::sqrt(std::max(0.0, discriminant));
        Scalar t1 = (-b + sqrtDiscriminant) / (2 * a);
        Scalar t2 = (-b - sqrtDiscriminant) / (2 * a);

        Vector3 intersection1 = linePoint1 + lineDir * t1;
        Vector3 intersection2 = linePoint1 + lineDir * t2;

        auto checkIntersection = [&](const Vector3& point) {
            Scalar projection = (point - cylinderBase).dot(cylAxisNorm);
            if (projection >= -EPSILON && projection <= height + EPSILON) {
                intersections.push_back(point);
            }
        };

        checkIntersection(intersection1);
        checkIntersection(intersection2);
    }

    // Check for intersection with the cylinder's caps
    // Bottom cap
    auto [intersectsBottomCap, bottomCapIntersection] = intersectLineDisk(
        linePoint1, linePoint2, cylinderBase, cylAxisNorm, radius);
    if (intersectsBottomCap) {
        Scalar projection = (bottomCapIntersection - cylinderBase).dot(cylAxisNorm);
        if (projection >= -EPSILON && projection <= EPSILON) {
            intersections.push_back(bottomCapIntersection);
        }
    }

    // Top cap
    Vector3 topCenter = cylinderBase + cylAxisNorm * height;
    auto [intersectsTopCap, topCapIntersection] = intersectLineDisk(
        linePoint1, linePoint2, topCenter, cylAxisNorm, radius);
    if (intersectsTopCap) {
        Scalar projection = (topCapIntersection - cylinderBase).dot(cylAxisNorm);
        if (projection >= height - EPSILON && projection <= height + EPSILON) {
            intersections.push_back(topCapIntersection);
        }
    }

    // If no intersections found, return empty vector (null result)
    return intersections;
}

/**
 * @brief Computes the intersection point of a line with a plane defined by a polygon.
 */
IntersectionResult intersectLinePolygon(
    const Vector3& linePoint1,
    const Vector3& linePoint2,
    const std::vector<Vector3>& polygonVertices)
{
    if (polygonVertices.size() < 3) {
        return std::make_tuple(false, Vector3::Zero());
    }

    Vector3 planeNormal = (polygonVertices[1] - polygonVertices[0]).cross(polygonVertices[2] - polygonVertices[0]).normalized();

    Vector3 lineDir = linePoint2 - linePoint1;
    Scalar denom = planeNormal.dot(lineDir);

    const Scalar EPSILON = std::numeric_limits<Scalar>::epsilon();

    if (std::abs(denom) < EPSILON) {
        return std::make_tuple(false, Vector3::Zero());
    }

    Scalar t = planeNormal.dot(polygonVertices[0] - linePoint1) / denom;
    Vector3 intersectionPoint = linePoint1 + lineDir * t;

    Vector3 absNormal = planeNormal.abs();
    int dominantAxis;
    if (absNormal.x > absNormal.y && absNormal.x > absNormal.z) {
        dominantAxis = 0; // x is the dominant axis, project onto yz-plane
    } else if (absNormal.y > absNormal.z) {
        dominantAxis = 1; // y is dominant, project onto xz-plane
    } else {
        dominantAxis = 2; // z is dominant, project onto xy-plane
    }

    std::vector<std::pair<double, double>> projectedPolygon;
    std::pair<double, double> projectedPoint;

    for (const auto& vertex : polygonVertices) {
        double u, v;
        double pu, pv;
        switch (dominantAxis) {
            case 0:
                u = vertex.y;
                v = vertex.z;
                pu = intersectionPoint.y;
                pv = intersectionPoint.z;
                break;
            case 1:
                u = vertex.x;
                v = vertex.z;
                pu = intersectionPoint.x;
                pv = intersectionPoint.z;
                break;
            case 2:
            default:
                u = vertex.x;
                v = vertex.y;
                pu = intersectionPoint.x;
                pv = intersectionPoint.y;
                break;
        }
        projectedPolygon.emplace_back(u, v);
        projectedPoint = std::make_pair(pu, pv);
    }

    // Point-in-polygon test using the ray casting algorithm
    bool inside = false;
    size_t n = projectedPolygon.size();
    for (size_t i = 0, j = n - 1; i < n; j = i++) {
        double yi = projectedPolygon[i].second;
        double yj = projectedPolygon[j].second;
        double xi = projectedPolygon[i].first;
        double xj = projectedPolygon[j].first;
        if (((yi > projectedPoint.second) != (yj > projectedPoint.second)) &&
            (projectedPoint.first < (xj - xi) * (projectedPoint.second - yi) / (yj - yi + EPSILON) + xi)) {
            inside = !inside;
        }
    }

    if (inside) {
        return std::make_tuple(true, intersectionPoint);
    } else {
        return std::make_tuple(false, Vector3::Zero());
    }
}

}  // namespace IntersectionLibrary

#endif  // INTERSECTION_LIBRARY_H
